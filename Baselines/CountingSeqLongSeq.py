import os

import numpy as np
import torch
import torch.nn.functional as F

from HOUDINI.Eval.CS_LS import TaskType, Dataset, get_task_name, \
    get_sequence_from_string, \
    mk_default_lib, get_task_settings, get_sequence_info, print_sequence
from HOUDINI.Eval.EvaluatorHelperFunctions import get_graph_a
from HOUDINI.Eval.EvaluatorUtils import get_io_examples_recognize_digit, \
    get_io_examples_count_digit_occ, get_io_examples_count_toys, get_io_examples_recognize_toy
# from HOUDINI.Eval.EvaluatorBaselinePNN_original_size import NetPNN, NetPNN_RNN
from HOUDINI.Interpreter.Interpreter import Interpreter, ProgramOutputType
from HOUDINI.Interpreter.NeuralModules import NetCNN as NetCNN_new, NetMLP as NetMLP_new, \
    NetRNN as NetRNN_new
from HOUDINI.FnLibraryFunctions import addImageFunctionsToLibrary
from HOUDINI.FnLibrary import FnLibrary

model_filepath_template = "{}/Models/{}.pth"


def get_model(task_info, dir_path, last_recogniser_ti=None, last_rnn_ti=None, load=False):
    """
    :param task_info:
    :param last_recogniser_ti: task info of the last recogniser. it's Null, if rnn appeared more recently.
    :param last_rnn_ti:
    :param dir_path:
    :param load:
    :return:
    """
    task_name = get_task_name(task_info)
    if task_info.task_type == TaskType.Recognise:
        cnn = NetCNN_new(task_name + "_cnn", 28, 1)
        mlp_cnn1 = NetMLP_new(task_name + "_mlp1_cnn_new", 1024, None, None, True)
        mlp_cnn2 = NetMLP_new(task_name + "_mlp2_cnn_new", 1024, 1, F.sigmoid, False)
    else:
        cnn = NetCNN_new(task_name + "_cnn", 28, 1)
        mlp_cnn1 = NetMLP_new(task_name + "_mlp1_cnn_new", 1024, None, None, True)
        mlp_cnn2 = NetMLP_new(task_name + "_mlp2_cnn_new", 1024, 1, F.sigmoid, False)

        rnn = NetRNN_new(task_name + "_rnn_new", 1, hidden_dim=100, output_dim=None)
        mlp_rnn = NetMLP_new(task_name + "_mlp_rnn_new", 100, 1, hidden_layer=False)

    if load:
        """
        Cases

        atm: CNN
        
        A) CNN  		(load CNN_cnn, CNN_mlp_cnn1)
        B) RNN  		(load RNN_cnn, RNN_mlp_cnn1)
        C) RNN, CNN	    (load CNN_cnn, CNN_mlp_cnn1)
        D) --CNN--, RNN	(load RNN_cnn, RNN_mlp_cnn1)
        
        
        atm: RNN
        
        E) CNN  		(load CNN_cnn, CNN_mlp_cnn1)
        F) RNN  		(load RNN_cnn, RNN_mlp_cnn1, RNN_mlp_cnn2, RNN_rnn)
        G) RNN, CNN	(load CNN_cnn, CNN_mlp_cnn1, RNN_rnn)
        H) --CNN--, RNN	(load RNN_cnn, RNN_mlp_cnn1, RNN_mlp_cnn2, RNN_rnn)
        
        
        """
        if task_info.task_type == TaskType.Recognise:
            if last_recogniser_ti is not None:
                last_rec_task_name = get_task_name(last_recogniser_ti)
                cnn.load(model_filepath_template.format(dir_path, last_rec_task_name + "_cnn"))
                mlp_cnn1.load(model_filepath_template.format(dir_path, last_rec_task_name + "_mlp1_cnn_new"))
            else:
                if last_rnn_ti is not None:
                    last_rnn_task_name = get_task_name(last_rnn_ti)
                    cnn.load(model_filepath_template.format(dir_path, last_rnn_task_name + "_cnn"))
                    mlp_cnn1.load(model_filepath_template.format(dir_path, last_rnn_task_name + "_mlp1_cnn_new"))

        else:  # RNN
            if last_recogniser_ti is not None: # case E
                last_rec_task_name = get_task_name(last_recogniser_ti)
                cnn.load(model_filepath_template.format(dir_path, last_rec_task_name + "_cnn"))
                mlp_cnn1.load(model_filepath_template.format(dir_path, last_rec_task_name + "_mlp1_cnn_new"))

                if last_rnn_ti is not None: # case G)
                    last_rnn_task_name = get_task_name(last_rnn_ti)
                    rnn.load(model_filepath_template.format(dir_path, last_rnn_task_name + "_rnn_new"))
            elif last_rnn_ti is not None: # cases F), H)
                last_rnn_task_name = get_task_name(last_rnn_ti)
                cnn.load(model_filepath_template.format(dir_path, last_rnn_task_name + "_cnn"))
                mlp_cnn1.load(model_filepath_template.format(dir_path, last_rnn_task_name + "_mlp1_cnn_new"))
                mlp_cnn2.load(model_filepath_template.format(dir_path, last_rnn_task_name + "_mlp2_cnn_new"))
                rnn.load(model_filepath_template.format(dir_path, last_rnn_task_name + "_rnn_new"))

    # return unkn_fn_dictionary, parameters, program
    if torch.cuda.is_available():
        cnn.cuda()
        mlp_cnn1.cuda()
        mlp_cnn2.cuda()
        if task_info.task_type == TaskType.Count:
            rnn.cuda()
            mlp_rnn.cuda()

    if task_info.task_type == TaskType.Recognise:
        program = "lambda inputs: {}({}({}(inputs)))".format(mlp_cnn2.name, mlp_cnn1.name, cnn.name)
        new_fns_dict = {cnn.name: cnn, mlp_cnn1.name: mlp_cnn1, mlp_cnn2.name: mlp_cnn2}
        parameters = list(cnn.parameters()) + list(mlp_cnn1.parameters()) + list(mlp_cnn2.parameters())
    else:
        program = "lambda inputs: {}({}(lib.map(lambda x: {}({}({}(x))), inputs)))".format(
            mlp_rnn.name, rnn.name, mlp_cnn2.name, mlp_cnn1.name, cnn.name)
        new_fns_dict = {cnn.name: cnn, mlp_cnn1.name: mlp_cnn1, mlp_cnn2.name: mlp_cnn2,
                        rnn.name: rnn, mlp_rnn.name: mlp_rnn}
        parameters = list(cnn.parameters()) + list(mlp_cnn1.parameters()) + list(mlp_cnn2.parameters()) +\
                     list(rnn.parameters()) + list(mlp_rnn.parameters())

    return program, new_fns_dict, parameters


def train_model(prefix, interpreter: Interpreter, io_examples_tr, io_examples_val, io_examples_test, task_info, save=False,
              dir_path=None, last_recogniser_ti=None, last_rnn_ti=None, load=False):
    if task_info.task_type == TaskType.Recognise:
        output_type = ProgramOutputType.SIGMOID
    else:
        output_type = ProgramOutputType.INTEGER

    data_loader_tr = interpreter._get_data_loader(io_examples_tr)
    data_loader_val = interpreter._get_data_loader(io_examples_val)
    data_loader_test = interpreter._get_data_loader(io_examples_test)

    program, new_fns_dict, parameters = get_model(task_info, dir_path, last_recogniser_ti, last_rnn_ti, load=load)

    new_fns_dict, max_accuracy_val, evaluations_np = interpreter.learn_neural_network_(program, output_type=output_type,
                                                                      new_fns_dict=new_fns_dict,
                                                                      trainable_parameters=list(parameters),
                                                                      data_loader_tr=data_loader_tr,
                                                                      data_loader_val=data_loader_val,
                                                                          data_loader_test=data_loader_test)

    max_accuracy_test = interpreter._get_accuracy(program, data_loader_test, output_type, new_fns_dict)
    print(max_accuracy_test)

    num_examples = io_examples_tr[0].shape[0] if type(io_examples_tr) == tuple else io_examples_tr[0][0].shape[0]

    np.save("{}/_{}__{}evaluations_np.npy".format(dir_path, prefix, num_examples), evaluations_np)
    if save:
        for key, value in new_fns_dict.items():
            value.save("{}/Models/".format(dir_path))

    return {"accuracy": max_accuracy_test}


def run_baseline_task(prefix, interpreter, task_settings, seq_tasks_info):
    load = settings["baseline_type"] == "wt"

    last_task_rnn = None # last counting task
    last_task_rnn_idx = -1
    last_task_cnn = None # recognition task
    last_task_cnn_idx = -1

    if seq_tasks_info.__len__() > 1:
        for task_idx in range(seq_tasks_info.__len__()-1):
            c_task = seq_tasks_info[task_idx]
            if c_task.task_type == TaskType.Recognise:
                last_task_cnn = c_task
                last_task_cnn_idx = task_idx
            else:
                last_task_rnn = c_task
                last_task_rnn_idx = task_idx

    if last_task_rnn is not None and last_task_cnn is not None:
        if last_task_rnn_idx > last_task_cnn_idx:
            last_task_cnn = None
            last_task_cnn_idx = -1

    task_info = seq_tasks_info[-1]
    # pnn = get_model(task_info, prev_models, dir_path, load=False)

    if task_info.task_type == TaskType.Recognise and task_info.dataset == Dataset.MNIST:
        io_examples_tr, io_examples_val, io_examples_test = get_io_examples_recognize_digit(task_info.index, task_settings.train_size, task_settings.val_size)
    elif task_info.task_type == TaskType.Recognise and task_info.dataset == Dataset.smallnorb:
        io_examples_tr, io_examples_val, io_examples_test = get_io_examples_recognize_toy(task_info.index, task_settings.train_size, task_settings.val_size)
    elif task_info.task_type == TaskType.Count and task_info.dataset == Dataset.MNIST:
        io_examples_tr, io_examples_val, io_examples_test = get_io_examples_count_digit_occ(task_info.index, task_settings.train_size, task_settings.val_size)
    elif task_info.task_type == TaskType.Count and task_info.dataset == Dataset.smallnorb:
        io_examples_tr, io_examples_val, io_examples_test = get_io_examples_count_toys(task_info.index, task_settings.train_size, task_settings.val_size)

    fn_parameters = {"interpreter": interpreter,
                     "io_examples_val": io_examples_val, "io_examples_test": io_examples_test,
                     "task_info": task_info, "dir_path": settings["results_dir"],
                     "last_recogniser_ti": last_task_cnn, "last_rnn_ti": last_task_rnn,
                     "load": load,
                     "prefix": prefix}

    get_graph_a(interpreter, task_settings.train_size, task_settings.training_percentages, io_examples_tr, train_model, fn_parameters, settings["results_dir"], prefix)
    # interpreter: Interpreter, io_examples_tr, io_examples_val, prev_models, task_info, save=False, dir_path=None


def main(task_id, sequence_str, sequence_name):
    # lib = mk_default_lib()
    lib = FnLibrary()
    addImageFunctionsToLibrary(lib, load_recognise_5s=False)

    task_settings = get_task_settings(settings["dbg_mode"], settings["dbg_learn_parameters"], synthesizer=None)

    seq_tasks_info = get_sequence_from_string(sequence_str)
    print("running task {} of the following sequence:".format(task_id + 1))
    print_sequence(seq_tasks_info)

    interpreter = Interpreter(lib, batch_size=150, epochs=task_settings.epochs)

    prefix = "{}_{}".format(sequence_name, task_id)

    run_baseline_task(prefix, interpreter, task_settings, seq_tasks_info[:task_id+1])


if __name__ == '__main__':
    settings = {
        "results_dir": "Results",
        "dbg_learn_parameters": True,  # If False, the interpreter doesn't learn the new parameters
        "dbg_mode": True,  # If True, the sequences run for a tiny amount of data

        "seq_string": "cs1",  # cs1, cs2, cs3, ls   # str(sys.argv[1])
        "baseline_type": "sa"  # "wt", "pnn"   #  str(sys.argv[2])
    }

    if not os.path.exists(settings["results_dir"]):
        os.makedirs(settings["results_dir"])

    seq_info_dict = get_sequence_info(settings["seq_string"])
    additional_prefix = "_{}".format(settings["baseline_type"])
    prefixes = ["{}{}".format(prefix, additional_prefix) for prefix in seq_info_dict["prefixes"]]

    for sequence_idx, sequence in enumerate(seq_info_dict["sequences"]):
        for task_id in range(seq_info_dict["num_tasks"]):
            main(task_id=task_id, sequence_str=sequence, sequence_name=prefixes[sequence_idx])
