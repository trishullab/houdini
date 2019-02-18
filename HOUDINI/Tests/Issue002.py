# How to derive from a named tuple

from HOUDINI.Eval.EvaluatorUtils import get_io_examples_classify_digits
from HOUDINI.Interpreter.Interpreter import Interpreter
from HOUDINI.FnLibrary import FnLibrary, PPLibItem
from HOUDINI.Synthesizer.AST import *


def main():
    tio, vio = get_io_examples_classify_digits(2000, 200)

    # Task Name: classify_digits
    prog = PPTermUnk(name='nn_fun_cs1cd_1', sort=PPFuncSort(args=[PPTensorSort(param_sort=PPReal(),
                                                                               shape=[PPDimConst(value=1),
                                                                                      PPDimConst(value=1),
                                                                                      PPDimConst(value=28),
                                                                                      PPDimConst(value=28)])],
                                                            rtpe=PPTensorSort(param_sort=PPBool(),
                                                                              shape=[PPDimConst(value=1),
                                                                                     PPDimConst(value=10)])))
    unkSortMap = {'nn_fun_cs1cd_1': PPFuncSort(args=[PPTensorSort(param_sort=PPReal(),
                                                                  shape=[PPDimConst(value=1), PPDimConst(value=1),
                                                                         PPDimConst(value=28), PPDimConst(value=28)])],
                                               rtpe=PPTensorSort(param_sort=PPBool(),
                                                                 shape=[PPDimConst(value=1), PPDimConst(value=10)]))}

    lib = FnLibrary()
    lib.addItems([PPLibItem(name='compose', sort=PPFuncSort(
        args=[PPFuncSort(args=[PPSortVar(name='B')], rtpe=PPSortVar(name='C')),
              PPFuncSort(args=[PPSortVar(name='A')], rtpe=PPSortVar(name='B'))],
        rtpe=PPFuncSort(args=[PPSortVar(name='A')], rtpe=PPSortVar(name='C'))), obj=None), PPLibItem(name='repeat',
                                                                                                     sort=PPFuncSort(
                                                                                                         args=[
                                                                                                             PPEnumSort(
                                                                                                                 start=8,
                                                                                                                 end=10),
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
                                                                                   PPDimVar(name='a')])), obj=None)])
    fn_sort = PPFuncSort(args=[PPTensorSort(param_sort=PPReal(),
                                            shape=[PPDimConst(value=1), PPDimConst(value=1), PPDimConst(value=28),
                                                   PPDimConst(value=28)])],
                         rtpe=PPTensorSort(param_sort=PPBool(), shape=[PPDimConst(value=1), PPDimConst(value=10)]))

    interpreter = Interpreter(lib, 150)
    res = interpreter.evaluate(program=prog,
                               output_type_s=fn_sort.rtpe,
                               unkSortMap=unkSortMap,
                               io_examples_tr=tio,
                               io_examples_val=vio)


if __name__ == '__main__':
    main()
