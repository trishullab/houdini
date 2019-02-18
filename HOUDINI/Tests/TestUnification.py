from HOUDINI.Synthesizer.AST import *
from HOUDINI.Synthesizer.ASTDSL import mkTensorSort
from HOUDINI.Synthesizer.Unification import unifyLists, applySubst


def test0():
    tensorA3 = mkTensorSort(PPInt(), ['A', 3])
    tensor2B = mkTensorSort(PPInt(), [2, 'B'])

    l1 = [tensorA3]
    l2 = [tensor2B]
    subst = unifyLists(l1, l2)
    print(subst)
    # [(PPDimVar(name='A'), PPDimConst(value=2)), (PPDimVar(name='B'), PPDimConst(value=3))]

    r1 = [applySubst(subst, x) for x in l1]
    r2 = [applySubst(subst, x) for x in l2]

    assert r1 == [PPTensorSort(param_sort=PPInt(), shape=[PPDimConst(value=2), PPDimConst(value=3)])]

    assert r1 == r2


def test1():
    listOfInt = PPListSort(PPInt())
    listOfT1 = PPListSort(PPSortVar('T1'))
    listOfT2 = PPListSort(PPSortVar('T2'))

    l1 = [listOfInt, listOfT1]
    l2 = [listOfInt, listOfT2]
    subst = unifyLists(l1, l2)
    print(subst)
    # [(PPSortVar(name='T1'), PPSortVar(name='T2'))]

    r1 = [applySubst(subst, x) for x in l1]
    r2 = [applySubst(subst, x) for x in l2]

    assert r1 == [PPListSort(param_sort=PPInt()), PPListSort(param_sort=PPSortVar(name='T2'))]

    assert r1 == r2


def test2():
    listOfA = PPListSort(PPSortVar('A'))
    listOfB = PPListSort(PPSortVar('B'))
    listOfT1 = PPListSort(PPSortVar('T1'))
    listOfT2 = PPListSort(PPSortVar('T2'))

    l1 = [listOfA, listOfB]
    l2 = [listOfT1, listOfT2]
    subst = unifyLists(l1, l2)
    print(subst)
    # [(PPSortVar(name='A'), PPSortVar(name='T1')), (PPSortVar(name='B'), PPSortVar(name='T2'))]

    r1 = [applySubst(subst, x) for x in l1]
    r2 = [applySubst(subst, x) for x in l2]

    assert r1 == [PPListSort(param_sort=PPSortVar(name='T1')), PPListSort(param_sort=PPSortVar(name='T2'))]

    assert r1 == r2


def test3():
    graphOfA = PPGraphSort(PPSortVar('A'))
    graphOfB = PPGraphSort(PPSortVar('B'))
    graphOfT1 = PPGraphSort(PPSortVar('T1'))
    graphOfT2 = PPGraphSort(PPSortVar('T2'))

    l1 = [graphOfA, graphOfB]
    l2 = [graphOfT1, graphOfT2]
    subst = unifyLists(l1, l2)
    print(subst)
    # [(PPSortVar(name='A'), PPSortVar(name='T1')), (PPSortVar(name='B'), PPSortVar(name='T2'))]

    r1 = [applySubst(subst, x) for x in l1]
    r2 = [applySubst(subst, x) for x in l2]

    assert r1 == [PPGraphSort(param_sort=PPSortVar(name='T1')), PPGraphSort(param_sort=PPSortVar(name='T2'))]

    assert r1 == r2
