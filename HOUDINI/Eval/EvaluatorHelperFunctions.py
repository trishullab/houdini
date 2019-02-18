import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from HOUDINI.Interpreter.Interpreter import Interpreter, ProgramOutputType
from HOUDINI.FnLibrary import PPLibItem
from Data import DataProvider
import os


def display_graph_a(label_to_tuple_of_numpy_map, xlabel="Training Dataset Size", ylabel="Accuracy after training",
                    negate_y=False, one_minus_y=False, results_directory="Eval/Results", savefilename=None):
    """
    :param list_of_tuples: (filename, name in the plot)
    :return:
    """
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    handles = []  # handles for the plt.legent method
    for label, np_tuple in label_to_tuple_of_numpy_map.items():
        x_axis_entries = np_tuple[0]
        if x_axis_entries.max() < 101:
            x_axis_entries = x_axis_entries * 60

        y_axis_entries = np_tuple[1]

        new_x_axis = []
        tpm_means = []
        tpm_std = []
        # First, try to correct for the Nones
        for idx in range(x_axis_entries.__len__()):
            not_none_values = []
            for y_idx in range(y_axis_entries.shape[1]):
                if y_axis_entries[idx][y_idx] is not None:
                    if negate_y:
                        y_axis_entries[idx][y_idx] = y_axis_entries[idx][y_idx] * (-1)
                    if one_minus_y:
                        y_axis_entries[idx][y_idx] = 1 - y_axis_entries[idx][y_idx]
                    not_none_values.append(y_axis_entries[idx][y_idx])
            if not_none_values.__len__() > 0:
                new_x_axis.append(x_axis_entries[idx])
                not_none_values_np = np.array(not_none_values)
                tpm_means.append(not_none_values_np.mean())
                tpm_std.append(not_none_values_np.std())

        tpm_means = np.array(tpm_means)
        tpm_std = np.array(tpm_std)

        plt.fill_between(new_x_axis, tpm_means - tpm_std,
                         tpm_means + tpm_std, alpha=0.1)  # , color=colors[i])
        t_line, = plt.plot(new_x_axis, tpm_means, 'o-', label=label, markersize=1, )

        handles.append(t_line)
    plt.legend(handles=handles)
    # plt.show(block=True)
    dir = "Eval/plots_may18"  # _longer_seq"
    dir = "Eval/plots_may18_longer_sequenes"

    if not os.path.exists(dir):
        os.makedirs(dir)

    plt.savefig("{}/{}.png".format(dir, savefilename))
    plt.clf()


def load_and_display_graph_a(dict_of_label_to_filename_list, results_directory="Eval/Results",
                             xlabel="Training Dataset Size", ylabel="Accuracy after training",
                             negate_y=False, one_minus_y=False, savefilename=None):
    label_to_tuple_of_numpy_map = {} # {"label": (numpy_x, numpy_y)}
    for label, filename_list in dict_of_label_to_filename_list.items():

        list_x = None
        list_y = None

        for filename in filename_list:
            # load the file:
            filepath = "{}/{}.npy".format(results_directory, filename)
            nparray = np.load(filepath)
            list_x = list_x if list_x is not None else nparray[0]

            c_list_y = nparray[1].reshape(-1, 1)
            list_y = c_list_y if list_y is None else np.hstack((list_y, c_list_y))

        label_to_tuple_of_numpy_map[label] = (list_x, list_y)

    # now that all files have been loaded, pass them to the real visualsiation function
    display_graph_a(label_to_tuple_of_numpy_map, results_directory=results_directory,
                    xlabel=xlabel, ylabel=ylabel, negate_y=negate_y, one_minus_y=one_minus_y, savefilename=savefilename)


def iterate_diff_training_sizes(train_io_examples, training_data_percentages):
    # assuming all lengths are represented equally
    num_of_training_dp = train_io_examples[0][0].shape[0] if type(train_io_examples) == list else train_io_examples[0].shape[0]

    for percentage in training_data_percentages:
        c_num_items = (percentage*num_of_training_dp) // 100
        if type(train_io_examples) == list:
            c_tr_io_examples = [(t[0][:c_num_items], t[1][:c_num_items]) for t in train_io_examples]
            return_c_num_items = c_num_items * train_io_examples.__len__()
        else:
            c_tr_io_examples = (train_io_examples[0][:c_num_items], train_io_examples[1][:c_num_items])
            return_c_num_items = c_num_items

        yield c_tr_io_examples, return_c_num_items, percentage == 100


def get_graph_a(interpreter, data_size_train, training_data_percentages, io_examples_tr, function: callable, fn_parameters: dict,
                results_directory=None, filename_to_save_the_plot_in=None):
    list_num_examples = []
    list_accuracies = []

    # train_size = io_examples_tr
    max_iterations = (data_size_train // interpreter.batch_size + (
        1 if data_size_train % interpreter.batch_size != 0 else 0)) * interpreter.original_num_epochs

    for c_tr_io_examples, c_num_items, is_100_percent in iterate_diff_training_sizes(io_examples_tr, training_data_percentages):

        c_iterations_per_epoch = c_num_items // interpreter.batch_size + (
            1 if c_num_items % interpreter.batch_size != 0 else 0)
        c_num_epochs = max_iterations // c_iterations_per_epoch + (
            1 if max_iterations % c_iterations_per_epoch != 0 else 0)
        # interpreter.epochs = c_num_epochs

        fn_parameters["io_examples_tr"] = c_tr_io_examples
        fn_parameters["save"] = is_100_percent
        result = function(**fn_parameters)

        list_accuracies.append(result["accuracy"])
        list_num_examples.append(c_num_items)

    if filename_to_save_the_plot_in is not None and results_directory is not None:
        nparray = np.array([list_num_examples, list_accuracies])
        np.save("{}/{}.npy".format(results_directory, filename_to_save_the_plot_in), nparray)

    """
    plt.plot(list_num_examples, list_accuracies)
    plt.ylabel('Accuracy')
    plt.xlabel('# Training Examples')
    plt.show(block=True)
    """

