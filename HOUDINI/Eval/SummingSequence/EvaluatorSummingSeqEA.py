
from HOUDINI.Eval.SummingSequence.EvaluatorSummingSeq import SummingSeqOne, SummingSeqTwo
from HOUDINI.Eval.Task import TaskSettings
from HOUDINI.Eval.TaskSeq import TaskSeqSettings
from HOUDINI.FnLibrary import FnLibrary
from HOUDINI.FnLibraryFunctions import get_items_from_repo
from HOUDINI.Synthesizer import GenUtils
from HOUDINI.Synthesizer.MiscUtils import setup_logging


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
        results_dir = "Results_summing_ea{}".format(repeat_id)
        seq_settings = TaskSeqSettings(
            update_library=True,
            results_dir=results_dir,
        )

        for task_id in [0, 1]:  # [0, 1]:
            if task_id == 0:
                N, M, K = 100, 1, 1
            else:
                N, M, K = 100000, 50, 50

            if not debug_mode:
                task_settings = TaskSettings(
                    train_size=6000,
                    val_size=2100,
                    training_percentages=[2, 10, 20, 50, 100],
                    N=N,
                    M=M,
                    K=K,
                    epochs=30,
                    synthesizer="evolutionary",
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
                    synthesizer="evolutionary",
                    debug_mode=debug_mode
                )

            seq = mkSeq(seq_id)
            seq.run(task_id)
            # seq.write_report(task_id)

if __name__ == '__main__':
    setup_logging()
    main()
    print("Done !")
