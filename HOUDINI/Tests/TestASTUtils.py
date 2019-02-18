from HOUDINI.FnLibraryFunctions import get_items_from_repo
from HOUDINI.FnLibrary import FnLibrary, PPLibItem
from HOUDINI.Synthesizer import ASTUtils
from HOUDINI.Synthesizer.ASTDSL import *
from HOUDINI.Synthesizer.ASTUtils import isAbstract, inferType, alphaConvertSorts
from HOUDINI.Synthesizer.ReprUtils import repr_py_sort


def testAbstract():
    sort = PPListSort(PPSortVar('T1'))
    res = isAbstract(sort)
    assert res is True

    sort = PPGraphSort(PPSortVar('T1'))
    res = isAbstract(sort)
    assert res is True

    sort = mkTensorSort(PPInt(), [1, 2, 'c'])
    res = isAbstract(sort)
    assert res is True

    sort = mkTensorSort(PPSortVar('T'), [1, 2, 3])
    res = isAbstract(sort)
    assert res is True

    sort = mkFuncSort(PPInt(), PPInt(), PPSortVar('T'), PPReal())
    res = isAbstract(sort)
    assert res is True


def testConcrete():
    sort = mkFuncSort(PPInt(), PPInt())
    res = isAbstract(sort)
    assert res is False

    sort = mkTensorSort(PPInt(), [1, 2, 3])
    res = isAbstract(sort)
    assert res is False

    sort = mkFuncSort(PPInt(), PPInt(), PPBool(), PPReal())
    res = isAbstract(sort)
    assert res is False


def mkDefaultLib():
    lib = FnLibrary()
    lib.addItems(get_items_from_repo(['compose', 'repeat', 'map_l', 'fold_l', 'conv_l', 'zeros']))
    return lib


def addRecogFive(lib):
    inType = mkRealTensorSort([1, 1, 28, 28])
    outType = mkBoolTensorSort([1, 1])
    recogDigitType = mkFuncSort(inType, outType)
    lib.addItem(PPLibItem('recogFive', recogDigitType, None))


def testInfer1():
    lib = mkDefaultLib()

    # region #+ Add recogFive to library
    inType = mkRealTensorSort([1, 1, 28, 28])
    outType = mkBoolTensorSort([1, 1])
    recogDigitType = mkFuncSort(inType, outType)

    lib.addItem(PPLibItem('recogFive', recogDigitType, None))
    # endregion

    inType = mkListSort(mkRealTensorSort([1, 1, 28, 28]))
    outType = mkRealTensorSort([1, 1])
    sort = mkFuncSort(inType, outType)

    prog = \
        PPFuncApp(fn=PPVar(name='lib.compose'),
                  args=[PPFuncApp(fn=PPVar(name='lib.fold_l'),
                                  args=[PPTermUnk(name='nn_fun_x_906',
                                                  sort=PPFuncSort(
                                                      args=[PPTensorSort(param_sort=PPReal(),
                                                                         shape=[PPDimConst(value=1),
                                                                                PPDimConst(value=1)]),
                                                            PPTensorSort(param_sort=PPBool(),
                                                                         shape=[PPDimConst(value=1),
                                                                                PPDimConst(value=1)])],
                                                      rtpe=PPTensorSort(param_sort=PPReal(),
                                                                        shape=[PPDimConst(value=1),
                                                                               PPDimConst(value=1)]))),
                                        PPFuncApp(fn=PPVar(name='lib.zeros'), args=[PPIntConst(value=1)])]),
                        PPFuncApp(fn=PPVar(name='lib.map_l'), args=[PPVar(name='lib.recogFive')])])

    sortInferred = inferType(prog, lib)

    print(repr_py_sort(sort))
    print(sort)

    print(repr_py_sort(sortInferred))
    print(sortInferred)

    assert sort == sortInferred


def testInfer2():
    lib = mkDefaultLib()

    sort = mkFuncSort(mkRealTensorSort([1, 1]), mkBoolTensorSort([1, 1]), mkRealTensorSort([1, 1]))

    prog = PPTermUnk(name='nn_fun_x_906',
                     sort=PPFuncSort(
                         args=[PPTensorSort(param_sort=PPReal(),
                                            shape=[PPDimConst(value=1),
                                                   PPDimConst(value=1)]),
                               PPTensorSort(param_sort=PPBool(),
                                            shape=[PPDimConst(value=1),
                                                   PPDimConst(value=1)])],
                         rtpe=PPTensorSort(param_sort=PPReal(),
                                           shape=[PPDimConst(value=1), PPDimConst(value=1)])))

    sortInferred = inferType(prog, lib)

    print(repr_py_sort(sort))
    print(sort)

    print(repr_py_sort(sortInferred))
    print(sortInferred)

    assert sort == sortInferred


def testInfer3():
    lib = mkDefaultLib()

    sort = mkRealTensorSort([1, 1])

    prog = PPFuncApp(fn=PPVar(name='lib.zeros'), args=[PPIntConst(value=1)])

    sortInferred = inferType(prog, lib)

    print(repr_py_sort(sort))
    print(sort)

    print(repr_py_sort(sortInferred))
    print(sortInferred)
    assert sort == sortInferred


def testInfer4():
    lib = mkDefaultLib()
    # region #+ Add recogFive to library
    inType = mkRealTensorSort([1, 1, 28, 28])
    outType = mkBoolTensorSort([1, 1])
    recogDigitType = mkFuncSort(inType, outType)

    lib.addItem(PPLibItem('recogFive', recogDigitType, None))
    # endregion

    inType = mkListSort(mkRealTensorSort([1, 1, 28, 28]))
    outType = mkListSort(mkBoolTensorSort([1, 1]))
    sort = mkFuncSort(inType, outType)

    prog = PPFuncApp(fn=PPVar(name='lib.map_l'), args=[PPVar(name='lib.recogFive')])

    sortInferred = inferType(prog, lib)

    print(repr_py_sort(sort))
    print(sort)

    print(repr_py_sort(sortInferred))
    print(sortInferred)

    assert sort == sortInferred


def testInfer5():
    lib = mkDefaultLib()

    # region #+ Add recogFive to library
    inType = mkRealTensorSort([1, 1, 28, 28])
    outType = mkBoolTensorSort([1, 1])
    recogDigitType = mkFuncSort(inType, outType)

    lib.addItem(PPLibItem('recogFive', recogDigitType, None))
    # endregion

    inType = mkRealTensorSort([1, 1, 28, 28])
    outType = mkRealTensorSort([1, 1])
    sort = mkFuncSort(inType, outType)

    prog = PPVar(name='lib.recogFive')

    sortInferred = inferType(prog, lib)

    print(repr_py_sort(sort))
    print(sort)

    print(repr_py_sort(sortInferred))
    print(sortInferred)

    assert sort == sortInferred


def testInfer6():
    lib = mkDefaultLib()

    inType = mkListSort(mkBoolTensorSort([1, 1]))
    outType = mkRealTensorSort([1, 1])
    sort = mkFuncSort(inType, outType)

    prog = PPFuncApp(fn=PPVar(name='lib.fold_l'),
                     args=[PPTermUnk(name='nn_fun_x_906',
                                     sort=PPFuncSort(
                                         args=[PPTensorSort(param_sort=PPReal(),
                                                            shape=[PPDimConst(value=1),
                                                                   PPDimConst(value=1)]),
                                               PPTensorSort(param_sort=PPBool(),
                                                            shape=[PPDimConst(value=1),
                                                                   PPDimConst(value=1)])],
                                         rtpe=PPTensorSort(param_sort=PPReal(),
                                                           shape=[PPDimConst(value=1), PPDimConst(value=1)]))),
                           PPFuncApp(fn=PPVar(name='lib.zeros'), args=[PPIntConst(value=1)])])

    sortInferred = inferType(prog, lib)

    print(repr_py_sort(sort))
    print(sort)

    print(repr_py_sort(sortInferred))
    print(sortInferred)

    assert sort == sortInferred


def testInfer7():
    lib = mkDefaultLib()

    # region #+ Add recogFive to library
    inType = mkRealTensorSort([1, 1, 28, 28])
    outType = mkBoolTensorSort([1, 1])
    recogDigitType = mkFuncSort(inType, outType)

    lib.addItem(PPLibItem('recogFive', recogDigitType, None))
    # endregion

    prog = PPFuncApp(fn=PPVar(name='lib.compose'),
                     args=[PPTermUnk(name='Unk',
                                     sort=PPFuncSort(args=[
                                         PPFuncSort(args=[PPListSort(param_sort=PPSortVar(name='A_1'))],
                                                    rtpe=PPListSort(param_sort=PPSortVar(name='B')))],
                                         rtpe=PPListSort(param_sort=PPTensorSort(param_sort=PPBool(),
                                                                                 shape=[PPDimConst(value=1),
                                                                                        PPDimConst(value=1)])))),
                           PPVar(name='lib.map_l')])

    sortInferred = inferType(prog, lib)
    assert sortInferred is None


def testInfer8():
    lib = mkDefaultLib()

    addRecogFive(lib)

    prog = PPVar(name='lib.map_l')
    sortInferred = inferType(prog, lib)
    print(repr_py_sort(sortInferred))
    # assert sortInferred is None




def testInfer9():
    lib = mkDefaultLib()

    addRecogFive(lib)

    sort = mkFuncSort(
        mkListSort(mkRealTensorSort([1, 1, 28, 28])),
        mkListSort(mkBoolTensorSort([1, 1])))

    prog = PPFuncApp(fn=PPVar(name='lib.map_l'),
              args=[
                  PPFuncApp(
                      fn=PPVar(name='lib.compose'),
                      args=[PPTermUnk(name='Unk',
                                      sort=PPFuncSort(args=[PPSortVar(name='C')],
                                                      rtpe=PPTensorSort(
                                                          param_sort=PPBool(),
                                                          shape=[PPDimConst(value=1),
                                                                 PPDimConst(
                                                                     value=1)]))),
                            PPFuncApp(fn=PPVar(
                                name='lib.compose'),
                                args=[PPTermUnk(
                                    name='Unk',
                                    sort=PPFuncSort(
                                        args=[
                                            PPTensorSort(
                                                param_sort=PPBool(),
                                                shape=[
                                                    PPDimConst(
                                                        value=1),
                                                    PPDimConst(
                                                        value=1)])],
                                        rtpe=PPSortVar(name='C'))),
                                    PPVar(name='lib.recogFive')])])])

    sortInferred = inferType(prog, lib)
    assert sort == sortInferred


def testInfer10():
    lib = mkDefaultLib()

    addRecogFive(lib)

    sort = mkFuncSort(
        mkListSort(mkRealTensorSort([1, 1, 28, 28])),
        mkListSort(mkBoolTensorSort([1, 1])))


    prog = PPFuncApp(
        fn=PPVar(
            name='lib.compose'),
        args=[PPFuncApp(fn=PPVar(name='lib.compose'),
                        args=[PPTermUnk(name='Unk',
                                        sort=PPFuncSort(
                                            args=[PPSortVar(name='B_1')],
                                            rtpe=PPListSort(
                                                param_sort=PPTensorSort(
                                                    param_sort=PPBool(),
                                                    shape=[PPDimConst(value=1),
                                                           PPDimConst(value=1)])))),
                              PPTermUnk(name='Unk',
                                        sort=PPFuncSort(
                                            args=[PPFuncSort(
                                                args=[PPListSort(
                                                    param_sort=PPSortVar(name='A'))],
                                                rtpe=PPListSort(
                                                    param_sort=PPSortVar(name='B_1_2')))],
                                            rtpe=PPSortVar(name='B_1')))]),
              PPFuncApp(fn=PPVar(name='lib.compose'),
                        args=[PPVar(name='lib.map_l'),
                              PPTermUnk(name='Unk',
                                        sort=PPFuncSort(
                                            args=[PPListSort(
                                                param_sort=PPTensorSort(param_sort=PPReal(),
                                                                        shape=[PPDimConst(value=1),
                                                                               PPDimConst(value=1),
                                                                               PPDimConst(value=28),
                                                                               PPDimConst(value=28)]))],
                                            rtpe=PPFuncSort(args=[PPSortVar(name='A')],
                                                            rtpe=PPSortVar(name='B_1_2'))))])])
    sortInferred = inferType(prog, lib)
    assert sort == sortInferred


def testInfer11():
    lib = mkDefaultLib()

    addRecogFive(lib)

    sort = mkFuncSort(
        mkListSort(mkRealTensorSort([1, 1, 28, 28])),
        mkListSort(mkBoolTensorSort([1, 1])))


    prog = PPFuncApp(
        fn=PPVar(
            name='lib.compose'),
        args=[PPFuncApp(fn=PPVar(name='lib.compose'),
                        args=[PPTermUnk(name='Unk',
                                        sort=PPFuncSort(
                                            args=[PPSortVar(name='B_1')],
                                            rtpe=PPListSort(
                                                param_sort=PPTensorSort(
                                                    param_sort=PPBool(),
                                                    shape=[PPDimConst(value=1),
                                                           PPDimConst(value=1)])))),
                              PPTermUnk(name='Unk',
                                        sort=PPFuncSort(
                                            args=[PPFuncSort(
                                                args=[PPListSort(
                                                    param_sort=PPSortVar(name='A'))],
                                                rtpe=PPListSort(
                                                    param_sort=PPSortVar(name='B_1_2')))],
                                            rtpe=PPSortVar(name='B_1')))]),
              PPFuncApp(fn=PPVar(name='lib.compose'),
                        args=[PPVar(name='lib.map_l'),
                              PPTermUnk(name='Unk',
                                        sort=PPFuncSort(
                                            args=[PPListSort(
                                                param_sort=PPTensorSort(param_sort=PPReal(),
                                                                        shape=[PPDimConst(value=1),
                                                                               PPDimConst(value=1),
                                                                               PPDimConst(value=28),
                                                                               PPDimConst(value=28)]))],
                                            rtpe=PPFuncSort(args=[PPSortVar(name='A')],
                                                            rtpe=PPSortVar(name='B_1_2'))))])])

    sortInferred = inferType(prog, lib)
    # prog is type checked but currently inferType only works for concrete types.
    assert sortInferred == sort


def testProgTreeSize():
    lib = mkDefaultLib()

    # region #+ Add recogFive to library
    inType = mkRealTensorSort([1, 1, 28, 28])
    outType = mkBoolTensorSort([1, 1])
    recogDigitType = mkFuncSort(inType, outType)

    lib.addItem(PPLibItem('recogFive', recogDigitType, None))
    # endregion

    inType = mkListSort(mkRealTensorSort([1, 1, 28, 28]))
    outType = mkRealTensorSort([1, 1])
    sort = mkFuncSort(inType, outType)

    prog = \
        PPFuncApp(fn=PPVar(name='lib.compose'),
                  args=[PPFuncApp(fn=PPVar(name='lib.fold_l'),
                                  args=[PPTermUnk(name='nn_fun_x_906',
                                                  sort=PPFuncSort(
                                                      args=[PPTensorSort(param_sort=PPReal(),
                                                                         shape=[PPDimConst(value=1),
                                                                                PPDimConst(value=1)]),
                                                            PPTensorSort(param_sort=PPBool(),
                                                                         shape=[PPDimConst(value=1),
                                                                                PPDimConst(value=1)])],
                                                      rtpe=PPTensorSort(param_sort=PPReal(),
                                                                        shape=[PPDimConst(value=1),
                                                                               PPDimConst(value=1)]))),
                                        PPFuncApp(fn=PPVar(name='lib.zeros'), args=[PPIntConst(value=1)])]),
                        PPFuncApp(fn=PPVar(name='lib.map_l'), args=[PPVar(name='lib.recogFive')])])

    size = ASTUtils.progTreeSize(prog)

    assert size == 7


def testAlphaConvertSorts():
    sort = mkFuncSort(
        mkListSort(mkRealTensorSort([1, 1, 28, 28])),
        mkListSort(mkBoolTensorSort([1, 1])))

    sortsToRename = [mkFuncSort(PPSortVar('A'), PPSortVar('B')),
                     mkFuncSort(PPSortVar('B'), PPSortVar('C')),
                     mkFuncSort(PPSortVar('C'), PPSortVar('A'))]

    sorts = [mkFuncSort(PPSortVar('D1'), PPSortVar('A')),
             mkFuncSort(PPSortVar('C'), PPSortVar('D2'))]

    newSortsToRename = alphaConvertSorts(sortsToRename, sorts)

    assert newSortsToRename[0] == mkFuncSort(PPSortVar('A0'), PPSortVar('B'))
    assert newSortsToRename[1] == mkFuncSort(PPSortVar('B'), PPSortVar('C0'))
    assert newSortsToRename[2] == mkFuncSort(PPSortVar('C0'), PPSortVar('A0'))



def testInfer12():
    # infeType fails on these programs
    lib = None
    # lib.compose(lib.compose(lib.compose(Unk, lib.compose(Unk, lib.compose(lib.map_l, lib.map_l))), Unk),
    # lib.map_l(lib.recogFive))
    prog1 = PPFuncApp(fn=PPVar(name='lib.compose'),
                     args=[
                         PPFuncApp(fn=PPVar(name='lib.compose'),args=[
                             PPFuncApp(fn=PPVar(name='lib.compose'),args=[
                                 PPTermUnk(name='Unk',
                                           sort=PPFuncSort(args=[PPSortVar(name='C')], rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))),
                                 PPFuncApp(fn=PPVar(name='lib.compose'), args=[
                                     PPTermUnk(name='Unk', sort=PPFuncSort(args=[PPFuncSort(args=[PPListSort(param_sort=PPListSort(param_sort=PPSortVar(name='A_1_2')))], rtpe=PPListSort(param_sort=PPListSort(param_sort=PPSortVar(name='B_1'))))], rtpe=PPSortVar(name='C'))),
                                     PPFuncApp(fn=PPVar(name='lib.compose'), args=[
                                         PPVar(name='lib.map_l'),
                                         PPVar(name='lib.map_l')])])]),
                             PPTermUnk(name='Unk', sort=PPFuncSort(args=[PPListSort(param_sort=PPTensorSort(
                                 param_sort=PPBool(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))], rtpe=PPFuncSort(args=[PPSortVar(name='A_1_2')], rtpe=PPSortVar(name='B_1'))))]),
                         PPFuncApp(fn=PPVar(name='lib.map_l'), args=[
                             PPVar(name='lib.recogFive')])])

    # lib.compose(Unk, lib.compose(lib.compose(lib.compose(lib.map_l, lib.map_l), lib.map_l), Unk))
    prog2 = PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPTermUnk(name='Unk', sort=PPFuncSort(args=[PPFuncSort(args=[
        PPListSort(param_sort=PPListSort(param_sort=PPListSort(param_sort=PPSortVar(name='A_1'))))], rtpe=PPListSort(param_sort=PPListSort(param_sort=PPListSort(param_sort=PPSortVar(name='B')))))], rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))), PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPVar(name='lib.map_l'), PPVar(name='lib.map_l')]), PPVar(name='lib.map_l')]), PPTermUnk(name='Unk', sort=PPFuncSort(args=[PPListSort(param_sort=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1), PPDimConst(value=28), PPDimConst(value=28)]))], rtpe=PPFuncSort(args=[PPSortVar(name='A_1')], rtpe=PPSortVar(name='B'))))])])

    # lib.compose(lib.compose(Unk, lib.compose(lib.map_l, lib.compose(lib.map_l, Unk))), lib.map_l(lib.recogFive))
    prog3 = PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPTermUnk(
        name='Unk', sort=PPFuncSort(args=[PPFuncSort(args=[PPListSort(param_sort=PPListSort(param_sort=PPSortVar(name='A_1')))], rtpe=PPListSort(param_sort=PPListSort(param_sort=PPSortVar(name='B_1_2'))))], rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))), PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPVar(name='lib.map_l'), PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPVar(name='lib.map_l'), PPTermUnk(name='Unk', sort=PPFuncSort(args=[PPListSort(param_sort=PPTensorSort(param_sort=PPBool(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))], rtpe=PPFuncSort(args=[PPSortVar(name='A_1')], rtpe=PPSortVar(name='B_1_2'))))])])]), PPFuncApp(fn=PPVar(name='lib.map_l'), args=[PPVar(name='lib.recogFive')])])

    # lib.compose(lib.fold_l(nn_fun_x_906, lib.zeros(1)), lib.compose(lib.compose(Unk, lib.compose(lib.map_l,
    # lib.compose(lib.map_l, lib.map_l))), Unk))
    prog4 = PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPFuncApp(fn=PPVar(name='lib.fold_l'), args=[PPTermUnk(
        name='nn_fun_x_906', sort=PPFuncSort(args=[PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]), PPTensorSort(param_sort=PPBool(), shape=[PPDimConst(value=1), PPDimConst(value=1)])], rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))), PPFuncApp(fn=PPVar(name='lib.zeros'), args=[PPIntConst(value=1)])]), PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPTermUnk(name='Unk', sort=PPFuncSort(args=[PPFuncSort(args=[PPListSort(param_sort=PPListSort(param_sort=PPListSort(param_sort=PPSortVar(name='A'))))], rtpe=PPListSort(param_sort=PPListSort(param_sort=PPListSort(param_sort=PPSortVar(name='B')))))], rtpe=PPListSort(param_sort=PPTensorSort(param_sort=PPBool(), shape=[PPDimConst(value=1), PPDimConst(value=1)])))), PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPVar(name='lib.map_l'), PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPVar(name='lib.map_l'), PPVar(name='lib.map_l')])])]), PPTermUnk(name='Unk', sort=PPFuncSort(args=[PPListSort(param_sort=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1), PPDimConst(value=28), PPDimConst(value=28)]))], rtpe=PPFuncSort(args=[PPSortVar(name='A')], rtpe=PPSortVar(name='B'))))])])


def testInfer12_1():
    lib = None
    # lib.compose(lib.compose(lib.compose(Unk, lib.compose(Unk, lib.compose(lib.map_l, lib.map_l))), Unk),
    # lib.map_l(lib.recogFive))
    prog1 = PPFuncApp(fn=PPVar(name='lib.compose'),
                     args=[
                         PPFuncApp(fn=PPVar(name='lib.compose'),args=[
                             PPFuncApp(fn=PPVar(name='lib.compose'),args=[
                                 PPTermUnk(name='Unk',
                                           sort=PPFuncSort(args=[PPSortVar(name='C')], rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))),
                                 PPFuncApp(fn=PPVar(name='lib.compose'), args=[
                                     PPTermUnk(name='Unk', sort=PPFuncSort(args=[PPFuncSort(args=[PPListSort(param_sort=PPListSort(param_sort=PPSortVar(name='A_1_2')))], rtpe=PPListSort(param_sort=PPListSort(param_sort=PPSortVar(name='B_1'))))], rtpe=PPSortVar(name='C'))),
                                     PPFuncApp(fn=PPVar(name='lib.compose'), args=[
                                         PPVar(name='lib.map_l'),
                                         PPVar(name='lib.map_l')])])]),
                             PPTermUnk(name='Unk', sort=PPFuncSort(args=[PPListSort(param_sort=PPTensorSort(
                                 param_sort=PPBool(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))], rtpe=PPFuncSort(args=[PPSortVar(name='A_1_2')], rtpe=PPSortVar(name='B_1'))))]),
                         PPFuncApp(fn=PPVar(name='lib.map_l'), args=[
                             PPVar(name='lib.recogFive')])])

    prog1 = PPFuncApp(fn=PPVar(name='lib.map_l'), args=[
                             PPVar(name='lib.recogFive')])
    lib = mkDefaultLib()

    addRecogFive(lib)

    sortInferred = inferType(prog1, lib)

    print(sortInferred)

    sort = PPFuncSort(args=[PPFuncSort(args=[
        PPListSort(param_sort=PPListSort(param_sort=PPListSort(param_sort=PPSortVar(name='A_1'))))], rtpe=PPListSort(param_sort=PPListSort(param_sort=PPListSort(param_sort=PPSortVar(name='B')))))], rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))

    print(repr_py_sort(sort))


def testInfer13():

    # lib.repeat(10, Unk)
    prog = PPFuncApp(fn=PPVar(name='lib.repeat'), args=[PPIntConst(value=10), PPTermUnk(name='Unk', sort=PPFuncSort(
        args=[
        PPListSort(param_sort=PPTensorSort(param_sort=PPReal(),
                                           shape=[PPDimConst(value=1), PPDimConst(value=1), PPDimConst(value=28),
                                                  PPDimConst(value=28)]))], rtpe=PPListSort(
        param_sort=PPTensorSort(param_sort=PPReal(),
                                shape=[PPDimConst(value=1), PPDimConst(value=1), PPDimConst(value=28),
                                       PPDimConst(value=28)]))))])

    prog = PPFuncApp(fn=PPVar(name='lib.repeat'), args=[PPIntConst(value=10), PPTermUnk(name='Unk', sort=PPFuncSort(
        args=[PPInt()], rtpe=PPInt()))])

    lib = mkDefaultLib()
    sortInferred = inferType(prog, lib)
    print(sortInferred)