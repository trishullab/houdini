import os
import sys

from HOUDINI.Eval.EvaluatorUtils import get_io_examples_regress_speed_mnist, \
    get_io_examples_shortest_path_mnist_maze, get_io_examples_shortest_path_street_sign_maze
from HOUDINI.Interpreter.Interpreter import Interpreter, ProgramOutputType
from HOUDINI.FnLibraryFunctions import get_items_from_repo
from HOUDINI.FnLibrary import FnLibrary


# model_filepath_template = 'Eval/ResultsNav/{}.pth'


def _train_s2t1(results_directory, save=True):
    train_io_examples, val_io_examples, test_io_examples = get_io_examples_regress_speed_mnist()

    name_cnn = "s2t1_cnn"
    name_mlp = "s2t1_mlp"

    program_output_type = ProgramOutputType.INTEGER
    nn_classifier_params1 = {"type": "CNN", "name": name_cnn, "input_dim": 28, "input_ch": 3}
    nn_classifier_params2 = {"type": "MLP", "name": name_mlp, "input_dim": 1024,
                             "output_dim": 2, "output_activation": None}

    program_train = "lambda inputs: {}({}(inputs))".format(name_mlp, name_cnn)
    new_fns = [nn_classifier_params1, nn_classifier_params2]
    result = interpreter.evaluate_(program=program_train, output_type=program_output_type,
                                   unknown_fns_def=new_fns, io_examples_tr=train_io_examples,
                                   io_examples_val=val_io_examples, io_examples_test=test_io_examples)

    if save:
        result["new_fns_dict"][name_cnn].save(results_directory)
        result["new_fns_dict"][name_mlp].save(results_directory)

    return result


def _train_s2t2(results_directory, save_name_cnn, save_name_mlp, save=True):
    if just_testing:
        io_train, io_val, io_test = get_io_examples_shortest_path_mnist_maze(150, 150, 150, 150)
    else:
        io_train, io_val, io_test = get_io_examples_shortest_path_mnist_maze(150, 70000, 1000000, 10000)

    name_cnn = "s2t2_cnn"
    name_mlp = "s2t2_mlp"
    name_conv_g = "s2t2_conv_g"

    load_from_cnn = "{}/{}.pth".format(results_directory, save_name_cnn)
    load_from_mlp = "{}/{}.pth".format(results_directory, save_name_mlp)

    nn_classifier_params1 = {"type": "CNN", "name": name_cnn, "input_dim": 28, "input_ch": 3,
                             "initialize_from": load_from_cnn}
    nn_classifier_params2 = {"type": "MLP", "name": name_mlp, "input_dim": 1024,
                             "output_dim": 2, "output_activation": None, "initialize_from": load_from_mlp}
    nn_conv_g_params = {"type": "GCONVNew", "name": name_conv_g, "input_dim": 2}

    new_fns_params = [nn_classifier_params1, nn_classifier_params2, nn_conv_g_params]
    program = "lib.compose(lib.repeat(10, lib.conv_g({})), lib.map_g(lib.compose({}, {})))".format(name_conv_g, name_mlp, name_cnn)

    result = interpreter.evaluate_(program, ProgramOutputType.INTEGER, new_fns_params, io_train, io_val, io_test,
                                   dont_train=False)

    if save:
        result["new_fns_dict"][name_cnn].save(results_directory)
        result["new_fns_dict"][name_mlp].save(results_directory)
        result["new_fns_dict"][name_conv_g].save(results_directory)
    return result


def _train_s2t3(results_directory, save_name_cnn, save_name_mlp, save_name_conv_g, save=True):
    if just_testing:
        io_train, io_val, io_test = get_io_examples_shortest_path_street_sign_maze(150, 150, 150, 150)
    else:
        io_train, io_val, io_test = get_io_examples_shortest_path_street_sign_maze(150, 70000, 1000000, 10000)

    name_cnn = "s2t3_cnn"
    name_mlp = "s2t3_mlp"
    name_conv_g = "s2t3_conv_g"

    load_from_cnn = "{}/{}.pth".format(results_directory, save_name_cnn)
    load_from_mlp = "{}/{}.pth".format(results_directory, save_name_mlp)
    load_from_conv_g = "{}/{}.pth".format(results_directory, save_name_conv_g)

    nn_classifier_params1 = {"type": "CNN", "name": name_cnn, "input_dim": 28, "input_ch": 3,
                             "initialize_from": load_from_cnn}
    nn_classifier_params2 = {"type": "MLP", "name": name_mlp, "input_dim": 1024,
                             "output_dim": 2, "output_activation": None, "initialize_from": load_from_mlp}
    nn_conv_g_params = {"type": "GCONVNew", "name": name_conv_g, "input_dim": 2, "initialize_from": load_from_conv_g}

    new_fns_params = [nn_classifier_params1, nn_classifier_params2, nn_conv_g_params]
    program = "lib.compose(lib.repeat(10, lib.conv_g({})), lib.map_g(lib.compose({}, {})))".format(name_conv_g, name_mlp, name_cnn)

    result = interpreter.evaluate_(program, ProgramOutputType.INTEGER, new_fns_params, io_train, io_val, io_test,
                                   dont_train=False)

    if save:
        result["new_fns_dict"][name_cnn].save(results_directory)
        result["new_fns_dict"][name_mlp].save(results_directory)
        result["new_fns_dict"][name_conv_g].save(results_directory)
    return result

if __name__ == '__main__':
    results_dir = str(sys.argv[1])
    # results_dir = "Results_maze_baselines"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    just_testing = False
    epochs_cnn = 1  # 000
    epochs_nav = 10
    batch_size = 150

    lib = FnLibrary()
    lib.addItems(get_items_from_repo(['flatten_2d_list', 'map_g', 'compose', 'repeat', 'conv_g']))

    interpreter = Interpreter(lib, epochs=1, batch_size=batch_size)
    # interpreter.epochs = epochs_cnn
    # res1 = _train_s2t1(results_dir)
    # print("res1: {}".format(res1["accuracy"]))

    interpreter.epochs = epochs_nav
    res2 = _train_s2t2(results_dir, "s2t1_cnn", "s2t1_mlp")
    print("res2: {}".format(res2["accuracy"]))

    interpreter.epochs = epochs_nav
    res3 = _train_s2t3(results_dir, "s2t2_cnn", "s2t2_mlp", "s2t2_conv_g")
    print("res3: {}".format(res3["accuracy"]))
