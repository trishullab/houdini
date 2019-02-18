from HOUDINI.Synthesizer import ReprUtils
from HOUDINI.Synthesizer.AST import PPFuncApp, PPTermUnk, PPSortVar, PPInt, PPReal, PPVar, PPFuncSort, \
    PPTensorSort, PPDimConst, PPListSort
from HOUDINI.Synthesizer.ASTUtils import isAbstract
from HOUDINI.Synthesizer.IntermediateTypeHandler import instantiateSortVar


def test_1():
    prog1 = PPFuncApp(
        fn=PPVar(name='lib.compose'),
        args=[
            PPTermUnk(name='nn_fun_csc4_2',
                      sort=PPFuncSort(
                          args=[
                              PPSortVar(name='B')],
                          rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))),
            PPTermUnk(name='nn_fun_csc4_3',
                      sort=PPFuncSort(
                          args=[PPListSort(param_sort=PPTensorSort(param_sort=PPReal(),
                                                                   shape=[PPDimConst(value=1), PPDimConst(value=1),
                                                                          PPDimConst(value=28),
                                                                          PPDimConst(value=28)]))],
                          rtpe=PPSortVar(name='B')))])

    prog2 = PPFuncApp(
        fn=PPVar(name='lib.compose'),
        args=[
            PPTermUnk(name='nn_fun_csc4_4',
                      sort=PPFuncSort(
                          args=[
                              PPSortVar(name='C')],
                          rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))),
            PPTermUnk(name='nn_fun_csc4_5',
                      sort=PPFuncSort(
                          args=[PPListSort(param_sort=PPTensorSort(param_sort=PPReal(),
                                                                   shape=[PPDimConst(value=1), PPDimConst(value=1),
                                                                          PPDimConst(value=28),
                                                                          PPDimConst(value=28)]))],
                          rtpe=PPSortVar(name='C')))])

    prog3 = PPFuncApp(
        fn=PPVar(name='lib.compose'),
        args=[prog1, prog2])

    cmts = [PPInt(), PPReal()]
    eprogs = instantiateSortVar(prog3, cmts, 2)
    assert len(eprogs) == 4
    for i, eprog in enumerate(eprogs):
        print(i, ReprUtils.repr_py(eprog))
        print(i, eprog)
        assert not isAbstract(eprog)
