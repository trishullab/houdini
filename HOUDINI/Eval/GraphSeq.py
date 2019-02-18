import argparse
from HOUDINI.Eval.EvaluatorUtils import \
    get_io_examples_regress_speed_mnist, \
    get_io_examples_regress_speed_street_sign, get_io_examples_shortest_path_street_sign_maze, \
    get_io_examples_shortest_path_mnist_maze  # get_io_examples_classify_speed, get_io_examples_shortest_path_speed_maze
from HOUDINI.Eval.Task import Task, TaskSettings
from HOUDINI.Eval.TaskSeq import TaskSeq, TaskSeqSettings
from HOUDINI.FnLibraryFunctions import get_items_from_repo
from HOUDINI.FnLibrary import FnLibrary
from HOUDINI.Synthesizer.ASTDSL import *
from HOUDINI.Synthesizer.MiscUtils import setup_logging


class RegressStreetTask(Task):
    def __init__(self, settings, seq):
        input_type = mkRealTensorSort([1, 3, 28, 28])
        output_type = mkRealTensorSort([1, 2])
        fn_sort = mkFuncSort(input_type, output_type)

        super(RegressStreetTask, self).__init__(fn_sort, settings, seq)

    def get_io_examples(self):
        return get_io_examples_regress_speed_street_sign()

    def name(self):
        return "regress_street"

    def sname(self):
        return "regstreet"


class RegressMNISTTask(Task):
    def __init__(self, settings, seq):
        input_type = mkRealTensorSort([1, 3, 28, 28])
        output_type = mkRealTensorSort([1, 2])
        fn_sort = mkFuncSort(input_type, output_type)

        super(RegressMNISTTask, self).__init__(fn_sort, settings, seq)

    def get_io_examples(self):
        return get_io_examples_regress_speed_mnist()

    def name(self):
        return "regress_mnist"

    def sname(self):
        return "regmnist"


class NavigateStreetTask(Task):
    def __init__(self, settings, seq, num3, num4, num5):
        input_type = mkGraphSort(mkRealTensorSort([1, 3, 28, 28]))
        output_type = mkGraphSort(mkRealTensorSort([1, 2]))
        fn_sort = mkFuncSort(input_type, output_type)

        super(NavigateStreetTask, self).__init__(fn_sort, settings, seq)
        self.num3 = num3
        self.num4 = num4
        self.num5 = num5

    def get_io_examples(self):
        return get_io_examples_shortest_path_street_sign_maze(150, self.num3, self.num4, self.num5)

    def name(self):
        return "navigate_street"

    def sname(self):
        return "navstreet"


class NavigateMNISTTask(Task):
    def __init__(self, settings, seq, num3, num4, num5):
        input_type = mkGraphSort(mkRealTensorSort([1, 3, 28, 28]))
        output_type = mkGraphSort(mkRealTensorSort([1, 2]))
        fn_sort = mkFuncSort(input_type, output_type)

        super(NavigateMNISTTask, self).__init__(fn_sort, settings, seq)
        self.num3 = num3
        self.num4 = num4
        self.num5 = num5

    def get_io_examples(self):
        return get_io_examples_shortest_path_mnist_maze(150, self.num3, self.num4, self.num5)

    def name(self):
        return "navigate_mnist"

    def sname(self):
        return "navmnist"


###########################
class GraphSeqOne(TaskSeq):
    def __init__(self, seq_settings, lib, dbg):

        if not dbg:
            num3, num4, num5 = 70000, 1000000, 10000
            num3b, num4b, num5b = 0, 12000, 10000
            N= 310000
        else:
            num3, num4, num5 = 150, 150, 150
            num3b, num4b, num5b = 0, 150, 150
            N = 1000

        regress_street_task_settings = TaskSettings(
            train_size=0,
            val_size=0,
            training_percentages=[100],
            N=10,
            M=10,
            K=10,
            epochs=1000,
            synthesizer=settings["synthesizer"],
            batch_size=10,
            dbg_learn_parameters=settings["dbg_learn_parameters"]
        )

        navigate_street_task_settings = TaskSettings(
            train_size=0,
            val_size=0,
            training_percentages=[100],
            N=N,
            M=60,
            K=10,
            epochs=10,
            synthesizer=settings["synthesizer"],
            batch_size=10,
            dbg_learn_parameters=settings["dbg_learn_parameters"]
        )

        if not dbg:
            num3, num4, num5 = 70000, 1000000, 10000
        else:
            num3, num4, num5 = 70, 1000, 10

        tasks = [
            RegressStreetTask(regress_street_task_settings, self),
            NavigateStreetTask(navigate_street_task_settings, self, num3=num3, num4=num4, num5=num5),
        ]

        super(GraphSeqOne, self).__init__(tasks, seq_settings, lib)

    def name(self):
        return 'graph_seq_1'

    def sname(self):
        return 'gs1'


class GraphSeqTwo(TaskSeq):
    def __init__(self, seq_settings, lib, dbg):

        if not dbg:
            num3, num4, num5 = 70000, 1000000, 10000
            num3b, num4b, num5b = 0, 12000, 10000
            N= 310000
        else:
            num3, num4, num5 = 150, 150, 150
            num3b, num4b, num5b = 0, 150, 150
            N = 1000

        regress_mnist_task_settings = TaskSettings(
            train_size=0,
            val_size=0,
            training_percentages=[100],
            N=10,
            M=10,
            K=10,
            epochs=1000,
            synthesizer=settings["synthesizer"],
            batch_size=10,
            dbg_learn_parameters=settings["dbg_learn_parameters"]
        )

        navigate_mnist_task_settings = TaskSettings(
            train_size=0,
            val_size=0,
            training_percentages=[100],
            N=N,
            M=60,  # 500
            K=10,
            epochs=10,
            synthesizer=settings["synthesizer"],
            batch_size=10,
            dbg_learn_parameters=settings["dbg_learn_parameters"]
        )

        navigate_street_task_settings = TaskSettings(
            train_size=0,
            val_size=0,
            training_percentages=[100],
            N=N,
            M=60,
            K=20,
            epochs=30,
            synthesizer=settings["synthesizer"],
            batch_size=10,
            dbg_learn_parameters=settings["dbg_learn_parameters"]
        )


        tasks = [
            RegressMNISTTask(regress_mnist_task_settings, self),
            NavigateMNISTTask(navigate_mnist_task_settings, self, num3=num3, num4=num4, num5=num5),
            NavigateStreetTask(navigate_street_task_settings, self, num3=num3b, num4=num4b, num5=num5b),
        ]

        super(GraphSeqTwo, self).__init__(tasks, seq_settings, lib)

    def name(self):
        return 'graph_seq_2'

    def sname(self):
        return 'gs2'


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--synthesizer',
                        choices=['enumerative', 'evolutionary'],
                        default='enumerative',
                        help='Synthesizer type. (default: %(default)s)')
    parser.add_argument('--taskseq',
                        choices=['gs1', 'gs2'],
                        required=True,
                        help='Task Sequence')
    parser.add_argument('--dbg',
                        action='store_true',
                        help='If set, the sequences run for a tiny amount of data'
                        )
    args = parser.parse_args()

    return args


def main():
    seq_id = 0 if settings["seq_string"] == "gs1" else 1
    num_tasks = 2 if seq_id == 0 else 3

    seq_settings = TaskSeqSettings(
        update_library=True,
        results_dir=settings["results_dir"],
    )

    def mkDefaultLib():
        lib = FnLibrary()
        lib.addItems(get_items_from_repo(['compose',
                                          # 'map_l', 'fold_l', 'conv_l',
                                          'conv_g', 'map_g', 'fold_g',
                                          'zeros', 'repeat']))
        return lib

    def mkSeq(seq_id_in):
        lib = mkDefaultLib()
        if seq_id_in == 0:
            return GraphSeqOne(seq_settings, lib, settings["dbg_mode"])
        elif seq_id_in == 1:
            return GraphSeqTwo(seq_settings, lib, settings["dbg_mode"])

    seq = mkSeq(seq_id)
    for task_id in range(num_tasks):
        seq.run(task_id)
    # seq.write_report(task_id)


if __name__ == '__main__':
    args = parse_args()

    settings = {
        "results_dir": "Results",
        "dbg_mode": args.dbg,  # If True, the sequences run for a tiny amount of data
        "dbg_learn_parameters": False,
        "synthesizer": args.synthesizer,  # enumerative, evolutionary
        "seq_string": args.taskseq  # "gs1, gs2"
    }

    setup_logging()
    main()
    print("Done !")
