# Summing sequence
# Sequence 1:
# classify_digits, sum_digits
#
# Sequence 2:
# sum_digits, classify_digits
#

from HOUDINI.Eval.EvaluatorUtils import get_io_examples_classify_digits, get_io_examples_sum_digits
from HOUDINI.Eval.Task import Task, TaskSettings
from HOUDINI.Eval.TaskSeq import TaskSeq, TaskSeqSettings
from HOUDINI.FnLibrary import FnLibrary
from HOUDINI.FnLibraryFunctions import get_items_from_repo
from HOUDINI.Synthesizer import GenUtils
from HOUDINI.Synthesizer.ASTDSL import *
from HOUDINI.Synthesizer.MiscUtils import setup_logging


class ClassifyDigitsTask(Task):
    def __init__(self, settings, seq):
        input_type = mkRealTensorSort([1, 1, 28, 28])
        output_type = mkBoolTensorSort([1, 10])
        fn_sort = mkFuncSort(input_type, output_type)

        super(ClassifyDigitsTask, self).__init__(fn_sort, settings, seq)

    def get_io_examples(self):
        return get_io_examples_classify_digits(self.settings.train_size, self.settings.val_size)

    def name(self):
        return "classify_digits"

    def sname(self):
        return 'cd'


class SumDigitsTask(Task):
    def __init__(self, settings, seq):
        input_type = mkListSort(mkRealTensorSort([1, 1, 28, 28]))
        output_type = mkRealTensorSort([1, 1])
        fn_sort = mkFuncSort(input_type, output_type)

        super(SumDigitsTask, self).__init__(fn_sort, settings, seq)

    def get_io_examples(self):
        return get_io_examples_sum_digits(self.settings.train_size, self.settings.val_size)

    def name(self):
        return "sum_digits"

    def sname(self):
        return 'sd'


#############

class SummingSeqOne(TaskSeq):
    def __init__(self, seq_settings, task_settings, repeat_id, lib):
        self.repeat_id = repeat_id
        tasks = [
            ClassifyDigitsTask(task_settings, self),
            SumDigitsTask(task_settings, self),
        ]

        super(SummingSeqOne, self).__init__(tasks, seq_settings, lib)

    def name(self):
        return 'summing_seq_1'

    def sname(self):
        return 'ss1'


class SummingSeqTwo(TaskSeq):
    def __init__(self, seq_settings, task_settings, lib):
        tasks = [
            SumDigitsTask(task_settings, self),
            ClassifyDigitsTask(task_settings, self),
        ]

        super(SummingSeqTwo, self).__init__(tasks, seq_settings, lib)

    def name(self):
        return 'summing_seq_2'

    def sname(self):
        return 'ss2'


def main():
    # seq_id = 0  # int(sys.argv[1]) - 1
    # repeat_id = 0  # int(sys.argv[2]) - 1
    # task_id = 0  # int(sys.argv[3]) - 1
    debug_mode = False

    print("PYTHONPATH: ")
    print(GenUtils.getPythonPath())
    print("PATH: ")
    print(GenUtils.getPath())

    def mkDefaultLib():
        lib = FnLibrary()
        lib.addItems(get_items_from_repo(['compose', 'repeat', 'map_l', 'fold_l', 'conv_l', 'zeros']))
        return lib

    def mkSeq(seq_id_in):
        lib = mkDefaultLib()
        if seq_id_in == 0:
            return SummingSeqOne(seq_settings, task_settings, repeat_id, lib)
        elif seq_id_in == 1:
            return SummingSeqTwo(seq_settings, task_settings, repeat_id, lib)

    seq_id = 0
    for repeat_id in range(3):
        results_dir = "Results_summing_td{}".format(repeat_id)
        seq_settings = TaskSeqSettings(
            update_library=True,
            results_dir=results_dir,
        )

        for task_id in [0, 1]:  # [0, 1]:
            if task_id == 0:
                N, M, K = 100, 10, 10
            else:
                N, M, K = 100000, 100, 100

            if not debug_mode:
                task_settings = TaskSettings(
                    train_size=6000,
                    val_size=2100,
                    training_percentages=[2, 10, 20, 50, 100],
                    N=N,
                    M=M,
                    K=K,
                    epochs=30,
                    synthesizer="enumerative",
                    filepath_synthesizer_pickle="ss_programs{}.pkl".format(task_id),
                    filepath_files_to_skip="files_to_skip_ss_t{}.txt".format(task_id + 1),
                    debug_mode=debug_mode
                )
            else:
                task_settings = TaskSettings(
                    train_size=150,
                    val_size=150,
                    training_percentages=[100],
                    N=N,
                    M=M,
                    K=K,
                    epochs=1,
                    synthesizer="enumerative",
                    filepath_synthesizer_pickle="ss_programs{}.pkl".format(task_id),
                    filepath_files_to_skip="files_to_skip_ss_t{}.txt".format(task_id + 1),
                    debug_mode=debug_mode
                )

            seq = mkSeq(seq_id)
            seq.run(task_id)
            # seq.write_report(task_id)


if __name__ == '__main__':
    setup_logging()
    main()
    print("Done !")
