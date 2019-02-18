from HOUDINI.Eval.SummingSequence.EvaluatorSummingSeq import SummingSeqOne, SummingSeqTwo
from HOUDINI.Eval.Task import TaskSettings
from HOUDINI.Eval.TaskSeq import TaskSeqSettings
from HOUDINI.FnLibraryFunctions import get_items_from_repo
from HOUDINI.FnLibrary import FnLibrary
from HOUDINI.Synthesizer import GenUtils
from HOUDINI.Synthesizer.MiscUtils import setup_logging


def main():

    arg1, arg2, arg3 = 1, 1, 1
    seq_id = arg1 - 1
    repeat_id = arg2 - 1
    task_id = arg3 - 1

    print("PYTHONPATH: ")
    print(GenUtils.getPythonPath())
    print("PATH: ")
    print(GenUtils.getPath())

    seq_settings = TaskSeqSettings(
        update_library=True,
        results_dir='Results',
    )

    task_settings = TaskSettings(
        train_size=600,
        val_size=210,
        training_percentages=[50, 100],
        N=10000,
        M=20,
        K=5,
        epochs=1,
        synthesizer='evolutionary',
    )

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

    seq = mkSeq(seq_id)
    seq.run(task_id)
    seq.write_report(task_id)


if __name__ == '__main__':
    setup_logging()
    main()
    print("Done !")
