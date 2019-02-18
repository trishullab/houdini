import os

import torch
import torch.nn.functional as F

from Baselines.PNN import NetPNN, NetPNN_RNN
from HOUDINI.Eval.EvaluatorHelperFunctions import get_graph_a
from HOUDINI.Eval.EvaluatorUtils import get_io_examples_classify_digits, get_io_examples_sum_digits
# from HOUDINI.Eval.EvaluatorBaselinePNN_original_size import NetPNN, NetPNN_RNN
from HOUDINI.Interpreter.Interpreter import Interpreter, ProgramOutputType
from HOUDINI.Interpreter.NeuralModules import NetCNN as NetCNN_new, NetMLP as NetMLP_new, \
    NetRNN as NetRNN_new
from HOUDINI.FnLibraryFunctions import addImageFunctionsToLibrary
from HOUDINI.FnLibrary import FnLibrary

model_filepath_template = "{}/Models/{}.pth"


def get_model_recogniser():
    cnn = NetCNN_new("ssr_cnn", 28, 1)
    mlp_cnn1 = NetMLP_new("ssr_mlp1_cnn", 1024, None, None, True)
    mlp_cnn2 = NetMLP_new("ssr_mlp2_cnn", 1024, 10, F.softmax, False)

    if torch.cuda.is_available():
        cnn.cuda()
        mlp_cnn1.cuda()
        mlp_cnn2.cuda()

    program = "lambda inputs: {}({}({}(inputs)))".format(mlp_cnn2.name, mlp_cnn1.name, cnn.name)
    new_fns_dict = {cnn.name: cnn, mlp_cnn1.name: mlp_cnn1, mlp_cnn2.name: mlp_cnn2}
    parameters = list(cnn.parameters()) + list(mlp_cnn1.parameters()) + list(mlp_cnn2.parameters())

    return program, new_fns_dict, parameters


def get_model_summer(dir_path, load):
    cnn = NetCNN_new("sss_cnn", 28, 1)
    mlp_cnn1 = NetMLP_new("sss_mlp1_cnn", 1024, None, None, True)
    mlp_cnn2 = NetMLP_new("sss_mlp2_cnn", 1024, 10, F.softmax, False)

    rnn = NetRNN_new("sss_rnn_new", 10, hidden_dim=100, output_dim=None)
    mlp_rnn = NetMLP_new("sss_mlp_rnn_new", 100, 1, hidden_layer=False)

    if load:
        cnn.load(model_filepath_template.format(dir_path, "ssr_cnn"))
        mlp_cnn1.load(model_filepath_template.format(dir_path, "ssr_mlp1_cnn"))

    if torch.cuda.is_available():
        cnn.cuda()
        mlp_cnn1.cuda()
        mlp_cnn2.cuda()
        rnn.cuda()
        mlp_rnn.cuda()

    program = "lambda inputs: {}({}(lib.map(lambda x: {}({}({}(x))), inputs)))".format(
        mlp_rnn.name, rnn.name, mlp_cnn2.name, mlp_cnn1.name, cnn.name)
    new_fns_dict = {cnn.name: cnn, mlp_cnn1.name: mlp_cnn1, mlp_cnn2.name: mlp_cnn2,
                    rnn.name: rnn, mlp_rnn.name: mlp_rnn}
    parameters = list(cnn.parameters()) + list(mlp_cnn1.parameters()) + list(mlp_cnn2.parameters()) + \
                 list(rnn.parameters()) + list(mlp_rnn.parameters())

    return program, new_fns_dict, parameters


def get_model_recogniser_pnn(dir_path=None, load=False):
    model = NetPNN("pnn_classifier", input_dim=28, input_ch=1, output_dim=10, output_activation=F.softmax,
                   past_models=None)

    if load:
        assert(dir_path is not None)
        model_filepath = model_filepath_template.format(dir_path, "pnn_classifier")
        model.load(model_filepath)

    if torch.cuda.is_available():
        model.cuda()

    new_fns_dict = {model.name : model}
    program = model.name
    parameters = model.parameters()

    return program, new_fns_dict, parameters


def get_model_summer_pnn(dir_path):
    _, past_model_dict, _ = get_model_recogniser_pnn(dir_path, load=True)
    past_model = list(past_model_dict.values())[0]
    model = NetPNN_RNN("pnn_summer", 28, 1, 10, F.softmax, 1, None, past_models=[past_model])

    if torch.cuda.is_available():
        model.cuda()

    new_fns_dict = {model.name: model}
    program = model.name
    parameters = model.parameters()

    return program, new_fns_dict, parameters


def train_classifier(type, interpreter: Interpreter, io_examples_tr, io_examples_val, io_examples_test, dir_path, save=False):
    output_type = ProgramOutputType.SOFTMAX
    if type=="sa" or type=="wt":
        program, new_fns_dict, parameters = get_model_recogniser()
    else:
        program, new_fns_dict, parameters = get_model_recogniser_pnn()

    # output_type = ProgramOutputType.INTEGER
    data_loader_tr = interpreter._get_data_loader(io_examples_tr)
    data_loader_val = interpreter._get_data_loader(io_examples_val)
    data_loader_test = interpreter._get_data_loader(io_examples_test)

    new_fns_dict, max_accuracy_val, _, evaluations_np = interpreter.learn_neural_network_(program, output_type=output_type,
                                                                      new_fns_dict=new_fns_dict,
                                                                      trainable_parameters=list(parameters),
                                                                      data_loader_tr=data_loader_tr,
                                                                      data_loader_val=data_loader_val,
                                                                          data_loader_test=data_loader_test)

    max_accuracy_test = interpreter._get_accuracy(program, data_loader_test, output_type, new_fns_dict)
    print(max_accuracy_test)

    # num_examples = io_examples_tr[0].shape[0] if type(io_examples_tr) == tuple else io_examples_tr[0][0].shape[0]
    # np.save("{}/_{}__{}evaluations_np.npy".format(dir_path, prefix, num_examples), evaluations_np)
    if save:
        for key, value in new_fns_dict.items():
            value.save("{}/Models/".format(dir_path))
    return {"accuracy": max_accuracy_test}


def train_summer(type, interpreter: Interpreter, io_examples_tr, io_examples_val, io_examples_test, dir_path, save=False):
    output_type = ProgramOutputType.INTEGER

    if type=="sa" or type=="wt":
        program, new_fns_dict, parameters = get_model_summer(dir_path, load=type=="wt")
    else:
        program, new_fns_dict, parameters = get_model_summer_pnn(dir_path)

    # output_type = ProgramOutputType.INTEGER
    data_loader_tr = interpreter._get_data_loader(io_examples_tr)
    data_loader_val = interpreter._get_data_loader(io_examples_val)
    data_loader_test = interpreter._get_data_loader(io_examples_test)

    new_fns_dict, max_accuracy_val, _, evaluations_np = interpreter.learn_neural_network_(program, output_type=output_type,
                                                                      new_fns_dict=new_fns_dict,
                                                                      trainable_parameters=list(parameters),
                                                                      data_loader_tr=data_loader_tr,
                                                                      data_loader_val=data_loader_val,
                                                                          data_loader_test=data_loader_test)

    max_accuracy_test = interpreter._get_accuracy(program, data_loader_test, output_type, new_fns_dict)
    print(max_accuracy_test)

    # num_examples = io_examples_tr[0].shape[0] if type(io_examples_tr) == tuple else io_examples_tr[0][0].shape[0]
    # np.save("{}/_{}__{}evaluations_np.npy".format(dir_path, prefix, num_examples), evaluations_np)
    if save:
        for key, value in new_fns_dict.items():
            value.save("{}/Models/".format(dir_path))
    return {"accuracy": max_accuracy_test}


def run_ss_classifier(interpreter, type, settings, dir_path):
    io_examples_tr, io_examples_val, io_examples_test = get_io_examples_classify_digits(settings["data_size_tr"], settings["data_size_val"])

    fn_parameters = { "type": type, "interpreter": interpreter,
                      "io_examples_val": io_examples_val, "io_examples_test": io_examples_test,
                      "dir_path": dir_path
                    }

    get_graph_a(interpreter, settings["data_size_tr"], settings["training_data_percentages"],
                io_examples_tr, train_classifier, fn_parameters, dir_path, "plot_classifier{}".format(type))


def run_ss_summer(interpreter, type, settings, dir_path):
    io_examples_tr, io_examples_val, io_examples_test = get_io_examples_sum_digits(settings["data_size_tr"],
                                                                                   settings["data_size_val"])
    fn_parameters = {"type": type, "interpreter": interpreter,
                     "io_examples_val": io_examples_val, "io_examples_test": io_examples_test,
                     "dir_path": dir_path
                     }

    get_graph_a(interpreter, settings["data_size_tr"], settings["training_data_percentages"],
                io_examples_tr, train_summer, fn_parameters, dir_path, "plot_summer{}".format(type))


def main(dir_path, type):
    if for_realz:
        settings = { "data_size_tr": 6000,
                     "data_size_val": 2100,
                     "num_epochs" : 30,
                     "training_data_percentages": [2, 10, 20, 50, 100] }
    else:
        settings = {"data_size_tr": 150,
                    "data_size_val": 150,
                    "num_epochs": 1,
                    "training_data_percentages": [100]}

    lib = FnLibrary()
    addImageFunctionsToLibrary(lib, load_recognise_5s=False)
    interpreter = Interpreter(lib, batch_size=150, epochs=settings["num_epochs"])

    run_ss_classifier(interpreter, type, settings, dir_path)
    run_ss_summer(interpreter, type, settings, dir_path)

if __name__ == '__main__':
    for_realz = True
    types = ["sa", "wt",  "pnn"]
    for type in types:
        for restart_idx in range(3):
            dir_path = "Results_ss_baselines{}{}".format(type, restart_idx)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            main(dir_path, type)
