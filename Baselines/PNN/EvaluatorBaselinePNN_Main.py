#from NeuralProgramming.Data.DataProvider import split_into_train_and_validation, get_batch_count_iseven, \
#    get_batch_count_var_len
from Data import *
from HOUDINI.Interpreter.Interpreter import Interpreter, ProgramOutputType
from HOUDINI.FnLibrary import FnLibrary
from HOUDINI.Interpreter.LibraryTensorFunctions import addImageFunctionsToLibrary

import torch

import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
# from HOUDINI.Evaluator.EvaluatorHelperFunctions import *
# from HOUDINI.Evaluator.EvaluatorBaselinePNN import NetPNN_RNN as NetPNN_RNN_slimmed
# from HOUDINI.Evaluator.EvaluatorBaselinePNN_original_size import NetPNN_RNN as NetPNN_RNN_org_size
import collections
import time
# from HOUDINI.Interpreter.NeuralModules import NetCNN as NetCNN_OG, NetMLP as NetMLP_OG, NetRNN as NetRNN_OG
from HOUDINI.Interpreter.NeuralModules import NetCNN as NetCNN_new, NetMLP as NetMLP_new, NetRNN as NetRNN_new


results_directory = "Eval/Results_Baselines_PNN"
model_filepath_template = "{}/{{}}.pth".format(results_directory)


def accuracy_test_np_model(interpreter, io_examples_tr, io_examples_val):
    data_loader_tr = interpreter._get_data_loader(io_examples_tr)
    data_loader_val = interpreter._get_data_loader(io_examples_val)

    output_type = ProgramOutputType.INTEGER

    cnn = NetCNN_new("cnn_new", 28, 1)
    mlp_cnn = NetMLP_new("mlp_cnn_new", 1024, 1, F.sigmoid, True)
    rnn = NetRNN_new("rnn_new", 1, 100, output_dim=1, output_activation=None)

    if torch.cuda.is_available():
        cnn.cuda()
        mlp_cnn.cuda()
        rnn.cuda()

    program = "lambda inputs: rnn_new(lib.map(lambda x: mlp_cnn_new(cnn_new(x)), inputs))"

    new_fns_dict = {"cnn_new": cnn, "mlp_cnn_new": mlp_cnn, "rnn_new": rnn}
    parameters = list(cnn.parameters()) + list(mlp_cnn.parameters()) + list(rnn.parameters())

    new_fns_dict, max_accuracy, _ = interpreter.learn_neural_network_(program, output_type=output_type,
                                                                      new_fns_dict=new_fns_dict,
                                                                      trainable_parameters=parameters,
                                                                      data_loader_tr=data_loader_tr,
                                                                      data_loader_val=data_loader_val)
    print(max_accuracy)
    return max_accuracy


def accuracy_test_baseline_model(interpreter, io_examples_tr, io_examples_val):
    data_loader_tr = interpreter._get_data_loader(io_examples_tr)
    data_loader_val = interpreter._get_data_loader(io_examples_val)

    output_type = ProgramOutputType.INTEGER

    task_name = "dtn" # dtn = dummy task name =)

    cnn = NetCNN_new(task_name + "_cnn", 28, 1)
    mlp_cnn1 = NetMLP_new(task_name + "_mlp1_cnn_new", 1024, None, None, True)
    mlp_cnn2 = NetMLP_new(task_name + "_mlp2_cnn_new", 1024, 1, F.sigmoid, False)
    rnn = NetRNN_new(task_name + "_rnn_new", 1, hidden_dim=100, output_dim=None)
    mlp_rnn = NetMLP_new(task_name + "_mlp_rnn_new", 100, 1, hidden_layer=False)

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

    new_fns_dict, max_accuracy, _ = interpreter.learn_neural_network_(program, output_type=output_type,
                                                                      new_fns_dict=new_fns_dict,
                                                                      trainable_parameters=parameters,
                                                                      data_loader_tr=data_loader_tr,
                                                                      data_loader_val=data_loader_val)
    print(max_accuracy)
    return max_accuracy


def accuracy_test_original_model(interpreter, io_examples_tr, io_examples_val):
    data_loader_tr = interpreter._get_data_loader(io_examples_tr)
    data_loader_val = interpreter._get_data_loader(io_examples_val)

    output_type = ProgramOutputType.INTEGER

    cnn = NetCNN_OG("cnn_og", 28, 1, None)
    mlp_cnn = NetMLP_OG("mlp_cnn_og", 1024, 1, None, output_activation=F.sigmoid)
    rnn = NetRNN_OG("rnn_og", 1, 100, None, None)
    mlp_rnn = NetMLP_OG("mlp_rnn_og", 100, 1, None, output_activation=None)

    if torch.cuda.is_available():
        cnn.cuda()
        mlp_cnn.cuda()
        rnn.cuda()
        mlp_rnn.cuda()

    program = "lambda inputs: mlp_rnn_og(rnn_og(lib.map(lambda x: mlp_cnn_og(cnn_og(x)), inputs)))"

    new_fns_dict = {"cnn_og": cnn, "mlp_cnn_og": mlp_cnn, "rnn_og": rnn, "mlp_rnn_og": mlp_rnn}
    parameters = list(cnn.parameters()) + list(mlp_cnn.parameters()) + list(rnn.parameters()) + list(mlp_rnn.parameters())

    new_fns_dict, max_accuracy, _ = interpreter.learn_neural_network_(program, output_type=output_type,
                                                                      new_fns_dict=new_fns_dict,
                                                                      trainable_parameters=parameters,
                                                                      data_loader_tr=data_loader_tr,
                                                                      data_loader_val=data_loader_val)
    print(max_accuracy)
    return max_accuracy


def accuracy_test_new_model(interpreter, io_examples_tr, io_examples_val):
    data_loader_tr = interpreter._get_data_loader(io_examples_tr)
    data_loader_val = interpreter._get_data_loader(io_examples_val)

    output_type = ProgramOutputType.INTEGER

    cnn = NetCNN_new("cnn_new", 28, 1)
    mlp_cnn = NetMLP_new("mlp_cnn_new", 1024, 1, F.sigmoid, True)
    rnn = NetRNN_new("rnn_new", 1, 100, output_dim=1, output_activation=None)

    if torch.cuda.is_available():
        cnn.cuda()
        mlp_cnn.cuda()
        rnn.cuda()

    program = "lambda inputs: rnn_new(lib.map(lambda x: mlp_cnn_new(cnn_new(x)), inputs))"

    new_fns_dict = {"cnn_new": cnn, "mlp_cnn_new": mlp_cnn, "rnn_new": rnn}
    parameters = list(cnn.parameters()) + list(mlp_cnn.parameters()) + list(rnn.parameters())

    new_fns_dict, max_accuracy, _ = interpreter.learn_neural_network_(program, output_type=output_type,
                                                                      new_fns_dict=new_fns_dict,
                                                                      trainable_parameters=parameters,
                                                                      data_loader_tr=data_loader_tr,
                                                                      data_loader_val=data_loader_val)
    print(max_accuracy)
    return max_accuracy

def speed_test_original_size(interpreter, io_examples_tr, io_examples_val):

    data_loader_tr = interpreter._get_data_loader(io_examples_tr)
    data_loader_val = interpreter._get_data_loader(io_examples_val)

    output_type = ProgramOutputType.INTEGER
    model_name = "pnn1"
    model = NetPNN_RNN_org_size(model_name, 28, 1, 1, F.sigmoid, 1, None, past_models=[])
    program = model_name

    new_fns_dict, max_accuracy, _ = interpreter.learn_neural_network_(program, output_type=output_type,
                                                                      new_fns_dict={model_name: model},
                                                                      trainable_parameters=list(model.parameters()),
                                                                      data_loader_tr=data_loader_tr,
                                                                      data_loader_val=data_loader_val)
    print("test1, original size, max_accuracy: {}".format(max_accuracy))


def speed_test_slimmed(interpreter, io_examples_tr, io_examples_val):
    data_loader_tr = interpreter._get_data_loader(io_examples_tr)
    data_loader_val = interpreter._get_data_loader(io_examples_val)

    output_type = ProgramOutputType.INTEGER
    model_name = "pnn1"
    model = NetPNN_RNN_slimmed(model_name, 28, 1, 1, F.sigmoid, 1, None, past_models=[])
    program = model_name

    new_fns_dict, max_accuracy, _ = interpreter.learn_neural_network_(program, output_type=output_type,
                                                                      new_fns_dict={model_name: model},
                                                                      trainable_parameters=list(model.parameters()),
                                                                      data_loader_tr=data_loader_tr,
                                                                      data_loader_val=data_loader_val)
    print("test1, slimmed size, max_accuracy: {}".format(max_accuracy))


def seq2(interpreter, mnist_data_provider, mnist_dict_train, mnist_dict_val):
    d1 = 5
    d2 = 5
    list_lengths_tr = [1, 2]
    list_lengths_val = [1, 2]
    t2_io_examples_tr = mnist_data_provider.get_batch_count_var_len([d1], 10000, mnist_dict_train,
                                                                      list_lengths=list_lengths_tr,
                                                                      return_count_int=False)
    t2_io_examples_val = mnist_data_provider.get_batch_count_var_len([d1], 1000, mnist_dict_val,
                                                                       list_lengths=list_lengths_val,
                                                                       return_count_int=False)
    data_loader_tr = interpreter._get_data_loader(t2_io_examples_tr)
    data_loader_val = interpreter._get_data_loader(t2_io_examples_val)

    t1_output_type = ProgramOutputType.INTEGER
    program1 = "pnn_rnn1"

    model1 = NetPNN("pnn1", 28, 1, 1, F.sigmoid, past_models=None)
    #model1.load(model_filepath_template.format("pnn1"))

    model2 = NetPNN("pnn2", 28, 1, 1, F.sigmoid, past_models=[model1])
    #model2.load(model_filepath_template.format("pnn1"))

    model3 = NetPNN_RNN("pnn_rnn1", 28, 1, 1, F.sigmoid, 1, None, past_models=[model1, model2])
    #model3.load(model_filepath_template.format("pnn_rnn1"))


    # model = NetPNN("pnn_rnn1", 28, 1, 1, F.sigmoid, past_models=None)
    # mode_parameters = list(model.parameters())
    # for m in mode_parameters:
    #    print(m.shape)
    new_fns_dict, max_accuracy, _ = interpreter.learn_neural_network_(program1, output_type=t1_output_type,
                                                                      new_fns_dict={"pnn_rnn1": model3},
                                                                      trainable_parameters=list(model3.parameters()),
                                                                      data_loader_tr=data_loader_tr,
                                                                      data_loader_val=data_loader_val)
    # print(max_accuracy)
    # new_fns_dict["pnn_rnn1"].save(results_directory)
    """
    model2 = NetPNN("pnn2", 28, 1, 1, F.sigmoid, past_models=[model1, model])
    t2_io_examples_tr = mnist_data_provider.get_batch_counting([d1], 1, 1000, mnist_dict_train,
                                                               single_items=True, return_count_int=False)
    t2_io_examples_val = mnist_data_provider.get_batch_counting([d1], 1, 10000, mnist_dict_val,
                                                                single_items=True, return_count_int=False)
    data_loader_tr = interpreter._get_data_loader(t2_io_examples_tr)
    data_loader_val = interpreter._get_data_loader(t2_io_examples_val)

    t2_output_type = ProgramOutputType.SIGMOID
    program1 = "pnn2"
    new_fns_dict, max_accuracy, _ = interpreter.learn_neural_network_(program1, output_type=t2_output_type,
                                                                      new_fns_dict={"pnn2": model2},
                                                                      trainable_parameters=list(model2.parameters()),
                                                                      data_loader_tr=data_loader_tr,
                                                                      data_loader_val=data_loader_val)
    print(max_accuracy)
    """


def seq1(interpreter, mnist_data_provider, mnist_dict_train, mnist_dict_val):
    d1 = 5
    d2 = 5
    """
    t1_io_examples_tr = mnist_data_provider.get_batch_counting([d1], 1, 1000, mnist_dict_train,
                                                                 single_items=True, return_count_int=False)
    t1_io_examples_val = mnist_data_provider.get_batch_counting([d1], 1, 10000, mnist_dict_val,
                                                                  single_items=True, return_count_int=False)

    data_loader_tr = interpreter._get_data_loader(t1_io_examples_tr)
    data_loader_val = interpreter._get_data_loader(t1_io_examples_val)

    t1_output_type = ProgramOutputType.SIGMOID
    program1 = "pnn1"
    model = NetPNN("pnn1", 28, 1, 1, F.sigmoid, past_models=None)
    # model.load(model_filepath_template.format("pnn1"))

    new_fns_dict, max_accuracy, _ = interpreter.learn_neural_network_(program1, output_type=t1_output_type,
                                                                       new_fns_dict={"pnn1": model},
                                                                       trainable_parameters=list(model.parameters()),
                                                                       data_loader_tr=data_loader_tr,
                                                                       data_loader_val=data_loader_val)
    print(max_accuracy)
    new_fns_dict["pnn1"].save(results_directory)
    """

    t2_io_examples_tr = mnist_data_provider.get_batch_counting([d1], 1, 1000, mnist_dict_train,
                                                               single_items=True, return_count_int=False)
    t2_io_examples_val = mnist_data_provider.get_batch_counting([d1], 1, 10000, mnist_dict_val,
                                                                single_items=True, return_count_int=False)
    data_loader_tr = interpreter._get_data_loader(t2_io_examples_tr)
    data_loader_val = interpreter._get_data_loader(t2_io_examples_val)
    
    model1 = NetPNN("pnn1", 28, 1, 1, F.sigmoid, past_models=None)
    model1.load(model_filepath_template.format("pnn1"))

    model2 = NetPNN("pnn2", 28, 1, 1, F.sigmoid, past_models=[model1])

    t2_output_type = ProgramOutputType.SIGMOID
    program1 = "pnn2"
    # model.load(model_filepath_template.format("pnn2"))

    new_fns_dict, max_accuracy, _ = interpreter.learn_neural_network_(program1, output_type=t2_output_type,
                                                                      new_fns_dict={"pnn2": model2},
                                                                      trainable_parameters=list(model2.parameters()),
                                                                      data_loader_tr=data_loader_tr,
                                                                      data_loader_val=data_loader_val)
    print(max_accuracy)
    # new_fns_dict["pnn1"].save(results_directory)



def main():
    for_realz = False
    if for_realz:
        data_size_tr = 6000
        data_size_val = 2100
        list_lengths_tr = [2, 3, 4, 5]
        list_lengths_val = [6, 7, 8]
        num_epochs = 20
    else:
        data_size_tr = 150  # 12000
        data_size_val = 150  # 2100
        list_lengths_tr = [1]  # [2, 3, 4, 5]
        list_lengths_val = [1]  # [6, 7, 8]
        num_epochs = 1  # 20


    lib = FnLibrary()
    addImageFunctionsToLibrary(lib, load_recognise_5s=False)
    interpreter = Interpreter(lib, batch_size=150, epochs=num_epochs)  #, evaluate_every_n_percent=70)  # 60)

    mnist_data_provider = MNISTDataProvider()
    mnist_dict_train, mnist_dict_val, mnist_dict_test = mnist_data_provider.split_into_train_and_validation(0, 12, shuffleFirst=True)

    d1 = 1
    io_examples_tr = mnist_data_provider.get_batch_count_var_len([d1], data_size_tr, mnist_dict_train,
                                                                 list_lengths=list_lengths_tr,
                                                                 return_count_int=False)
    io_examples_val = mnist_data_provider.get_batch_count_var_len([d1], data_size_val, mnist_dict_val,
                                                                  list_lengths=list_lengths_val,
                                                                  return_count_int=False)
    acc_np = []
    acc_baseline = []
    for i in range(10):
        acc_np.append(accuracy_test_np_model(interpreter, io_examples_tr, io_examples_val))
        acc_baseline.append(accuracy_test_baseline_model(interpreter, io_examples_tr, io_examples_val))

    print("NP average error: {}".format(sum(acc_np) / len(acc_np)))
    print("New average error: {}".format(sum(acc_baseline) / len(acc_baseline)))
    """
    start_time = time.clock()
    for i in range(10):
        speed_test_slimmed(interpreter, io_examples_tr, io_examples_val)
    time_slimmed = time.clock() - start_time

    start_time = time.clock()
    for i in range(10):
        speed_test_original_size(interpreter, io_examples_tr, io_examples_val)
    time_original = time.clock() - start_time

    print("time_slimmed: {} seconds".format(time_slimmed))
    print("time_original: {} seconds".format(time_original))
    """
    #seq1(interpreter, mnist_data_provider, mnist_dict_train, mnist_dict_val)
    #seq2(interpreter, mnist_data_provider, mnist_dict_train, mnist_dict_val)


if __name__ == '__main__':
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    main()
