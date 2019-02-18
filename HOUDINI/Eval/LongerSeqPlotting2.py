from Data import *
from HOUDINI.Interpreter.Interpreter import Interpreter, ProgramOutputType
from HOUDINI.FnLibrary import FnLibrary
from HOUDINI.Interpreter.LibraryTensorFunctions import addImageFunctionsToLibrary
# from HOUDINI.Interpreter.NeuralModules import *
# from HOUDINI.NewLibrary import PPLibItem
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from HOUDINI.Eval.EvaluatorHelperFunctions import *
import collections

import matplotlib.pyplot as plt
import numpy as np


def plot_sequence(task_id):
    label_sa = "standalone"
    label_wt = "low-level-transfer"
    label_pnn = "progressive-nn"
    label_np = "Houdini-TD"
    label_evol = "Houdini-Evol"

    c_graph_dict = collections.OrderedDict()
    c_graph_dict.update({label_sa: []})
    c_graph_dict.update({label_np: []})
    c_graph_dict.update({label_evol: []})
    c_graph_dict.update({label_pnn: []})
    c_graph_dict.update({label_wt: []})


    for i in range(5):
        c_graph_dict[label_sa].append("d_comb_{}_longer_seq_{}t{}_plot".format(i+1, "sa", task_id+ 1))
        c_graph_dict[label_np].append(tasks[task_id][i].format("np"))
        c_graph_dict[label_evol].append(tasks[task_id][i].format("evol"))
        c_graph_dict[label_pnn].append("d_comb_{}_longer_seq_{}t{}_plot".format(i+1, "pnn", task_id + 1))
        c_graph_dict[label_wt].append("d_comb_{}_longer_seq_{}t{}_plot".format(i+1, "wt", task_id + 1))


    if task_id in [0, 1, 4, 5, 6, 8]:
        ylabel = "RMSE"
        one_minus_y = False
        negate_y = True
    else:
        ylabel = "Classification Error"
        one_minus_y = True
        negate_y = False

    load_and_display_graph_a(c_graph_dict, results_directory=results_directory, ylabel=ylabel, negate_y=negate_y,
                             one_minus_y=one_minus_y, savefilename="t{}".format(task_id + 1))


def get_performance(model, task_id, x_axis_index):

    if task_id in [0, 1, 4, 5, 6, 8]:
        ylabel = "RMSE"
        one_minus_y = False
        negate_y = True
    else:
        ylabel = "Classification Error"
        one_minus_y = True
        negate_y = False


    values = []
    for i in range(5):
        if model == 0:
            filename = "d_comb_{}_longer_seq_{}t{}_plot".format(i+1, "sa", task_id+ 1)
        elif model == 1:
            filename = tasks[task_id][i].format("np")
        elif model == 2:
            filename = tasks[task_id][i].format("evol")
        elif model == 3:
            filename = "d_comb_{}_longer_seq_{}t{}_plot".format(i+1, "pnn", task_id + 1)
        elif model == 4:
            filename = "d_comb_{}_longer_seq_{}t{}_plot".format(i+1, "wt", task_id + 1)

        file = np.load("{}/{}.npy".format(results_directory, filename))
        if file[1][x_axis_index] is not None:
            v = file[1][x_axis_index]
            if one_minus_y:
                v = 1 - v
            if negate_y:
                v = v * (-1)
            values.append(v)

    return np.array(values).mean()


def plot_sequence1(task_ids, percentage, ylabel):

    label_sa = "standalone"
    label_wt = "low-level-transfer"
    label_pnn = "progressive-nn"
    label_np = "HOUDINI"
    label_evol = "evolutionary"

    x_axis_sa = []
    y_axis_sa = []

    x_axis_np = []
    y_axis_np = []

    x_axis_evol = []
    y_axis_evol = []

    x_axis_pnn = []
    y_axis_pnn = []

    x_axis_wt = []
    y_axis_wt = []

    for task_id in task_ids:
        c_perf_sa = get_performance(0, task_id, percentage)
        if c_perf_sa is not None:
            x_axis_sa.append(task_id)
            y_axis_sa.append(c_perf_sa)

        c_perf_np = get_performance(1, task_id, percentage)
        if c_perf_np is not None:
            x_axis_np.append(task_id)
            y_axis_np.append(c_perf_np)

        c_perf_evol = get_performance(2, task_id, percentage)
        if c_perf_evol is not None:
            x_axis_evol.append(task_id)
            y_axis_evol.append(c_perf_evol)

        c_perf_pnn = get_performance(3, task_id, percentage)
        if c_perf_pnn is not None:
            x_axis_pnn.append(task_id)
            y_axis_pnn.append(c_perf_pnn)

        c_perf_wt = get_performance(4, task_id, percentage)
        if c_perf_wt is not None:
            x_axis_wt.append(task_id)
            y_axis_wt.append(c_perf_wt)


    print(y_axis_sa)
    print(y_axis_np)
    print(y_axis_evol)
    print(y_axis_pnn)
    print(y_axis_wt)

    plt.scatter(x_axis_sa, y_axis_sa, s=50.5, label=label_sa, alpha=0.5)
    plt.scatter(x_axis_np, y_axis_np, s=50.5, label=label_np, alpha=0.5)
    plt.scatter(x_axis_evol, y_axis_evol, s=50.5, label=label_evol, alpha=0.5)
    plt.scatter(x_axis_pnn, y_axis_pnn, s=50.5, label=label_pnn, alpha=0.5)
    plt.scatter(x_axis_wt, y_axis_wt, s=50.5, label=label_wt, alpha=0.5)

    plt.legend()
    plt.xlabel("Task Index")
    plt.ylabel(ylabel)

    plt.ylim((0., 0.5))

    plt.show(block=True)


results_directory = "LongerSequences/Combined"
#for task_id in range(9):
#    plot_sequence(task_id)


t9 =   ["d_comb_1_longer_seq_{}/count_digit_occ_2s_plot",
        "d_comb_2_longer_seq_{}/count_digit_occ_8s_plot",
        "d_comb_3_longer_seq_{}/count_digit_occ_3s_plot",
        "d_comb_4_longer_seq_{}/count_digit_occ_5s_plot",
        "d_comb_5_longer_seq_{}/count_digit_occ_0s_plot"
        ]

t8 =   ["d_comb_1_longer_seq_{}/recognize_digit_9_plot",
        "d_comb_2_longer_seq_{}/recognize_digit_5_plot",
        "d_comb_3_longer_seq_{}/recognize_digit_1_plot",
        "d_comb_4_longer_seq_{}/recognize_digit_8_plot",
        "d_comb_5_longer_seq_{}/recognize_digit_7_plot"
        ]

t7 =   ["d_comb_1_longer_seq_{}/count_toys_0s_plot",
        "d_comb_2_longer_seq_{}/count_toys_1s_plot",
        "d_comb_3_longer_seq_{}/count_toys_2s_plot",
        "d_comb_4_longer_seq_{}/count_toys_3s_plot",
        "d_comb_5_longer_seq_{}/count_toys_4s_plot"
        ]


t6 =   ["d_comb_1_longer_seq_{}/count_digit_occ_9s_plot",
        "d_comb_2_longer_seq_{}/count_digit_occ_5s_plot",
        "d_comb_3_longer_seq_{}/count_digit_occ_1s_plot",
        "d_comb_4_longer_seq_{}/count_digit_occ_8s_plot",
        "d_comb_5_longer_seq_{}/count_digit_occ_7s_plot"
        ]

t5 =   ["d_comb_1_longer_seq_{}/count_toys_2s_plot",
        "d_comb_2_longer_seq_{}/count_toys_4s_plot",
        "d_comb_3_longer_seq_{}/count_toys_3s_plot",
        "d_comb_4_longer_seq_{}/count_toys_1s_plot",
        "d_comb_5_longer_seq_{}/count_toys_0s_plot"
        ]

t1 =   ["d_comb_1_longer_seq_{}/count_digit_occ_7s_plot",
        "d_comb_2_longer_seq_{}/count_digit_occ_1s_plot",
        "d_comb_3_longer_seq_{}/count_digit_occ_9s_plot",
        "d_comb_4_longer_seq_{}/count_digit_occ_6s_plot",
        "d_comb_5_longer_seq_{}/count_digit_occ_4s_plot"
        ]

t2 =   ["d_comb_1_longer_seq_{}/count_toys_4s_plot",
        "d_comb_2_longer_seq_{}/count_toys_0s_plot",
        "d_comb_3_longer_seq_{}/count_toys_1s_plot",
        "d_comb_4_longer_seq_{}/count_toys_2s_plot",
        "d_comb_5_longer_seq_{}/count_toys_3s_plot"
        ]

t3 =   ["d_comb_1_longer_seq_{}/recognize_toy_plot",
        "d_comb_2_longer_seq_{}/recognize_toy_plot",
        "d_comb_3_longer_seq_{}/recognize_toy_plot",
        "d_comb_4_longer_seq_{}/recognize_toy_plot",
        "d_comb_5_longer_seq_{}/recognize_toy_plot"
        ]

t4 =   ["d_comb_1_longer_seq_{}/recognize_digit_9_plot",
        "d_comb_2_longer_seq_{}/recognize_digit_5_plot",
        "d_comb_3_longer_seq_{}/recognize_digit_1_plot",
        "d_comb_4_longer_seq_{}/recognize_digit_8_plot",
        "d_comb_5_longer_seq_{}/recognize_digit_7_plot"
        ]

tasks = [t1, t2, t3, t4, t5, t6, t7, t8, t9]


#for task_index in range(9):
#    plot_sequence(task_index)

"""
["cntD7, cntT4, recT0, recD9, cntT2, cntD9, cntT0, recD7, cntD2",
 "cntD1, cntT0, recT1, recD5, cntT4, cntD5, cntT1, recD1, cntD8",
 "cntD9, cntT1, recT2, recD1, cntT3, cntD1, cntT2, recD9, cntD3",
 "cntD6, cntT2, recT3, recD8, cntT1, cntD8, cntT3, recD6, cntD5",
 "cntD4, cntT3, recT4, recD7, cntT0, cntD7, cntT4, recD4, cntD0"]
"""

plot_sequence1([0, 1, 4, 5, 6, 8], 0, "RMSE")
plot_sequence1([2, 3, 7], 0, "Classification Error")