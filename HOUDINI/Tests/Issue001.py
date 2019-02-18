from HOUDINI.Eval.EvaluatorUtils import get_io_examples_count_digit_occ
from HOUDINI.NeuralSynthesizer import NeuralSynthesizer
from HOUDINI.FnLibrary import FnLibrary, PPLibItem
from HOUDINI.Synthesizer import ReprUtils
from HOUDINI.Synthesizer.AST import *


def main():
    io_train, io_val = get_io_examples_count_digit_occ(5, 1200, 1200)
    # Task Name: count_digit_occ_5s
    prog = PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPTermUnk(name='nn_fun_3254', sort=PPFuncSort(args=[PPListSort(
        param_sort=PPListSort(
            param_sort=PPTensorSort(param_sort=PPBool(), shape=[PPDimConst(value=1), PPDimConst(value=1)])))],
        rtpe=PPTensorSort(
            param_sort=PPReal(),
            shape=[
                PPDimConst(
                    value=1),
                PPDimConst(
                    value=1)]))),
                                                         PPFuncApp(fn=PPVar(name='lib.conv_l'), args=[
                                                             PPFuncApp(fn=PPVar(name='lib.map_l'),
                                                                       args=[PPVar(name='lib.nn_fun_1')])])])
    unkSortMap = {'nn_fun_3254': PPFuncSort(args=[PPListSort(param_sort=PPListSort(
        param_sort=PPTensorSort(param_sort=PPBool(), shape=[PPDimConst(value=1), PPDimConst(value=1)])))],
        rtpe=PPTensorSort(param_sort=PPReal(),
                          shape=[PPDimConst(value=1), PPDimConst(value=1)]))}
    lib = FnLibrary()
    lib.addItems([PPLibItem(name='compose', sort=PPFuncSort(
        args=[PPFuncSort(args=[PPSortVar(name='B')], rtpe=PPSortVar(name='C')),
              PPFuncSort(args=[PPSortVar(name='A')], rtpe=PPSortVar(name='B'))],
        rtpe=PPFuncSort(args=[PPSortVar(name='A')], rtpe=PPSortVar(name='C'))), obj=None), PPLibItem(name='repeat',
                                                                                                     sort=PPFuncSort(
                                                                                                         args=[
                                                                                                             PPEnumSort(
                                                                                                                 start=2,
                                                                                                                 end=50),
                                                                                                             PPFuncSort(
                                                                                                                 args=[
                                                                                                                     PPSortVar(
                                                                                                                         name='A')],
                                                                                                                 rtpe=PPSortVar(
                                                                                                                     name='A'))],
                                                                                                         rtpe=PPFuncSort(
                                                                                                             args=[
                                                                                                                 PPSortVar(
                                                                                                                     name='A')],
                                                                                                             rtpe=PPSortVar(
                                                                                                                 name='A'))),
                                                                                                     obj=None),
                  PPLibItem(name='map_l',
                            sort=PPFuncSort(args=[PPFuncSort(args=[PPSortVar(name='A')], rtpe=PPSortVar(name='B'))],
                                            rtpe=PPFuncSort(args=[PPListSort(param_sort=PPSortVar(name='A'))],
                                                            rtpe=PPListSort(param_sort=PPSortVar(name='B')))),
                            obj=None), PPLibItem(name='fold_l', sort=PPFuncSort(
            args=[PPFuncSort(args=[PPSortVar(name='B'), PPSortVar(name='A')], rtpe=PPSortVar(name='B')),
                  PPSortVar(name='B')],
            rtpe=PPFuncSort(args=[PPListSort(param_sort=PPSortVar(name='A'))], rtpe=PPSortVar(name='B'))), obj=None),
                  PPLibItem(name='conv_l', sort=PPFuncSort(
                      args=[PPFuncSort(args=[PPListSort(param_sort=PPSortVar(name='A'))], rtpe=PPSortVar(name='B'))],
                      rtpe=PPFuncSort(args=[PPListSort(param_sort=PPSortVar(name='A'))],
                                      rtpe=PPListSort(param_sort=PPSortVar(name='B')))), obj=None),
                  PPLibItem(name='zeros', sort=PPFuncSort(args=[PPDimVar(name='a')],
                                                          rtpe=PPTensorSort(param_sort=PPReal(),
                                                                            shape=[PPDimConst(value=1),
                                                                                   PPDimVar(name='a')])), obj=None),
                  PPLibItem(name='nn_fun_1', sort=PPFuncSort(args=[PPTensorSort(param_sort=PPReal(),
                                                                                shape=[PPDimConst(value=1),
                                                                                       PPDimConst(value=1),
                                                                                       PPDimConst(value=28),
                                                                                       PPDimConst(value=28)])],
                                                             rtpe=PPTensorSort(param_sort=PPBool(),
                                                                               shape=[PPDimConst(value=1),
                                                                                      PPDimConst(value=1)])), obj=None),
                  PPLibItem(name='nn_fun_2230', sort=PPFuncSort(args=[PPTensorSort(param_sort=PPReal(),
                                                                                   shape=[PPDimConst(value=1),
                                                                                          PPDimConst(value=1),
                                                                                          PPDimConst(value=28),
                                                                                          PPDimConst(value=28)])],
                                                                rtpe=PPTensorSort(param_sort=PPBool(),
                                                                                  shape=[PPDimConst(value=1),
                                                                                         PPDimConst(value=1)])),
                            obj=None)])
    fn_sort = PPFuncSort(args=[PPListSort(param_sort=PPTensorSort(param_sort=PPReal(),
                                                                  shape=[PPDimConst(value=1), PPDimConst(value=1),
                                                                         PPDimConst(value=28), PPDimConst(value=28)]))],
                         rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))

    print(ReprUtils.repr_py_ann(prog))
    print(ReprUtils.repr_py_sort(lib.items['nn_fun_1'].sort))
    NeuralSynthesizer.is_evaluable(prog)


if __name__ == '__main__':
    main()
