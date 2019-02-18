import argparse
import sys
import random
from collections import namedtuple
from enum import Enum

from HOUDINI.Eval.CS_LS_Tasks import RecognizeDigitTask, RecognizeToyTask, \
    CountToysTask, CountDigitOccTask
from HOUDINI.Eval.Task import TaskSettings
from HOUDINI.Eval.TaskSeq import TaskSeqSettings, TaskSeq
from HOUDINI.FnLibraryFunctions import get_items_from_repo
from HOUDINI.FnLibrary import FnLibrary


class TaskType(Enum):
    Recognise = 1
    Count = 2


class Dataset(Enum):
    MNIST = 1
    smallnorb = 2


class_range_MNIST = range(10)
class_range_smallnorb = range(5)
SequenceTaskInfo = namedtuple("SequenceTaskInfo", ["task_type", "dataset", "index"])


def get_randomised_sequence(length):
    randomised_sequence = []

    for i in range(length):
        c_task_type = random.choice(list(TaskType))
        c_dataset = random.choice(list(Dataset))
        if c_dataset == Dataset.MNIST:
            c_index = random.choice(class_range_MNIST)
        else:
            c_index = random.choice(class_range_smallnorb)

        randomised_sequence.append(SequenceTaskInfo(task_type=c_task_type, dataset=c_dataset, index=c_index))

    return randomised_sequence


def get_task_name(task_info):
    if task_info.task_type == TaskType.Recognise:
        c_str = "rec"
    else:
        c_str = "cnt"

    if task_info.dataset == Dataset.MNIST:
        c_str += "D"
    else:
        c_str += "T"

    c_str += str(task_info.index)
    return c_str


def print_sequence(sequence):
    str_list = []
    for task_info in sequence:
        str_list.append(get_task_name(task_info))

    print("[ " + ", ".join(str_list) + " ]")


def get_sequence_from_string(sequence_str):
    """
    :param sequence_str: recD3, countT2, recT1, ...  (no initial/closing brackets)
    """
    sequence = []
    str_list = sequence_str.split(", ")
    for c_str in str_list:
        if c_str[:3] == "rec":
            c_task_type = TaskType.Recognise
        else:
            c_task_type = TaskType.Count

        if c_str[3:4] == "D":
            c_dataset = Dataset.MNIST
        else:
            c_dataset = Dataset.smallnorb

        c_index = int(c_str[4:5])

        sequence.append(SequenceTaskInfo(task_type=c_task_type, dataset=c_dataset, index=c_index))

    return sequence


class CountingSequence(TaskSeq):
    def __init__(self, name, list_task_info, dbg_learn_parameters, seq_settings, task_settings, lib):
        self._name = name
        tasks = []
        for tidx, task_info in enumerate(list_task_info):
            if task_info.task_type == TaskType.Recognise and task_info.dataset == Dataset.MNIST:
                tasks.append(RecognizeDigitTask(task_settings, task_info.index, self, dbg_learn_parameters))

            if task_info.task_type == TaskType.Recognise and task_info.dataset == Dataset.smallnorb:
                tasks.append(RecognizeToyTask(task_settings, task_info.index, self, dbg_learn_parameters))

            if task_info.task_type == TaskType.Count and task_info.dataset == Dataset.MNIST:
                tasks.append(CountDigitOccTask(task_settings, task_info.index, self, dbg_learn_parameters))

            if task_info.task_type == TaskType.Count and task_info.dataset == Dataset.smallnorb:
                tasks.append(CountToysTask(task_settings, task_info.index, self, dbg_learn_parameters))

        super(CountingSequence, self).__init__(tasks, seq_settings, lib)

    def name(self):
        return self._name  # 'counting_sequence'

    def sname(self):
        return self._name


def get_sequence_info(seq_string):
    cs1_idx_pairs = cs2_idx_pairs = [(0, 1), (3, 4), (5, 7), (2, 9), (6, 8)]
    cs3_idx_pairs = [(0, 1), (3, 2), (5, 3), (2, 0), (9, 4)]
    ls_idx_pairs = [(7, 4, 0, 9, 2, 9, 0, 7, 2),
                    (1, 0, 1, 5, 4, 5, 1, 1, 8),
                    (9, 1, 2, 1, 3, 1, 2, 9, 3),
                    (6, 2, 3, 8, 1, 8, 3, 6, 5),
                    (4, 3, 4, 7, 0, 7, 4, 4, 0)]

    cs1 = ["recD{0}, recD{1}, cntD{0}, cntD{1}".format(idx1, idx2) for idx1, idx2 in cs1_idx_pairs]
    cs2 = ["recD{0}, cntD{0}, cntD{1}, recD{1}".format(idx1, idx2) for idx1, idx2 in cs2_idx_pairs]
    cs3 = ["recD{0}, cntD{0}, cntT{1}, recT{1}".format(idx1, idx2) for idx1, idx2 in cs3_idx_pairs]
    ls = ["cntD{}, cntT{}, recT{}, recD{}, cntT{}, cntD{}, cntT{}, recD{}, cntD{}".format(*idx_tuple)
          for idx_tuple in ls_idx_pairs]

    cs1_prefixes = ["cs1_d{}d{}".format(idx1, idx2) for idx1, idx2 in cs1_idx_pairs]
    cs2_prefixes = ["cs2_d{}d{}".format(idx1, idx2) for idx1, idx2 in cs2_idx_pairs]
    cs3_prefixes = ["cs3_d{}t{}".format(idx1, idx2) for idx1, idx2 in cs3_idx_pairs]
    ls_prefixes = ["ls_d{}t{}".format(idx_tuple[0], idx_tuple[1]) for idx_tuple in ls_idx_pairs]

    dict = {
        "cs1": {"sequences": cs1, "prefixes": cs1_prefixes, "num_tasks": 4},
        "cs2": {"sequences": cs2, "prefixes": cs2_prefixes, "num_tasks": 4},
        "cs3": {"sequences": cs3, "prefixes": cs3_prefixes, "num_tasks": 4},
        "ls": {"sequences": ls, "prefixes": ls_prefixes, "num_tasks": 9}
    }

    assert (seq_string in dict.keys())
    return dict[seq_string]


def get_task_settings(dbg_mode, dbg_learn_parameters, synthesizer=None):
    """
    :param dbg_mode:
    :param synthesizer_type: None, enumerative, evolutionary
    :return:
    """
    if not dbg_mode:
        task_settings = TaskSettings(
            train_size=6000,
            val_size=2100,
            training_percentages=[2, 10, 20, 50, 100],
            N=10000,
            M=50,
            K=50,
            epochs=30,
            synthesizer=synthesizer,
            dbg_learn_parameters=dbg_learn_parameters
        )
    else:
        task_settings = TaskSettings(
            train_size=150,
            val_size=150,
            training_percentages=[100],
            N=1000,
            M=2,
            K=2,
            epochs=1,
            synthesizer=synthesizer,
            dbg_learn_parameters=dbg_learn_parameters
        )
    return task_settings


def mk_default_lib():
    lib = FnLibrary()
    lib.addItems(get_items_from_repo(['compose', 'repeat', 'map_l', 'fold_l', 'conv_l', 'zeros']))
    return lib


def main(task_id, sequence_str, sequence_name, synthesizer):
    seq_settings = TaskSeqSettings(
        update_library=True,
        results_dir=settings["results_dir"],
    )
    task_settings = get_task_settings(settings["dbg_mode"], settings["dbg_learn_parameters"], synthesizer=synthesizer)
    lib = mk_default_lib()

    seq_tasks_info = get_sequence_from_string(sequence_str)
    print_sequence(seq_tasks_info)

    seq = CountingSequence(sequence_name, seq_tasks_info, settings["dbg_learn_parameters"], seq_settings, task_settings,
                           lib)
    seq.run(task_id)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--synthesizer',
                        choices=['enumerative', 'evolutionary'],
                        default='enumerative',
                        help='Synthesizer type. (default: %(default)s)')
    parser.add_argument('--taskseq',
                        choices=['cs1', 'cs2', 'cs3', 'ls'],
                        required=True,
                        help='Task Sequence')
    parser.add_argument('--dbg',
                        action='store_true',
                        help='If set, the sequences run for a tiny amount of data'
                        )
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    settings = {
        "results_dir": "Results",  # str(sys.argv[1])
        "dbg_learn_parameters": True,  # If False, the interpreter doesn't learn the new parameters
        "dbg_mode": args.dbg,  # If True, the sequences run for a tiny amount of data
        "synthesizer": args.synthesizer,  # enumerative, evolutionary
        "seq_string": args.taskseq  # "ls"  # cs1, cs2, cs3, ls
    }

    seq_info_dict = get_sequence_info(settings["seq_string"])

    # num_tasks = seq_info_dict["num_tasks"]
    additional_prefix = "_np_{}".format("td" if settings["synthesizer"] == "enumerative" else "ea")
    prefixes = ["{}{}".format(prefix, additional_prefix) for prefix in seq_info_dict["prefixes"]]

    for sequence_idx, sequence in enumerate(seq_info_dict["sequences"]):
        for task_id in range(seq_info_dict["num_tasks"]):
            main(task_id=task_id, sequence_str=sequence,
                 sequence_name=prefixes[sequence_idx], synthesizer=settings["synthesizer"])
