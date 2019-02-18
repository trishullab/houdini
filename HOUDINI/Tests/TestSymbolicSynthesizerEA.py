import random

from HOUDINI.FnLibraryFunctions import get_items_from_repo
from HOUDINI.FnLibrary import FnLibrary, PPLibItem
from HOUDINI.Synthesizer.AST import PPFuncApp, PPVar, PPTermUnk, PPFuncSort, PPTensorSort, PPReal, PPDimConst, \
    PPBool, PPIntConst, PPTermNT, PPListSort, PPSortVar
from HOUDINI.Synthesizer.ASTDSL import mkRealTensorSort, mkBoolTensorSort, mkFuncSort, mkListSort
from HOUDINI.Synthesizer.ASTUtils import inferType
from HOUDINI.Synthesizer.ReprUtils import repr_py, repr_py_ann
from HOUDINI.Synthesizer.Rules import expandToUnk
from HOUDINI.Synthesizer.SymbolicSynthesizerEA import mutate, ProgramGenerator, getIdTermSorts, crossover, \
    SymbolicSynthesizerEA


def mkDefaultLib():
    lib = FnLibrary()
    lib.addItems(get_items_from_repo(['compose', 'repeat', 'map_l', 'fold_l', 'conv_l', 'zeros']))
    return lib


def addRecogFive(lib):
    inType = mkRealTensorSort([1, 1, 28, 28])
    outType = mkBoolTensorSort([1, 1])
    recogDigitType = mkFuncSort(inType, outType)
    lib.addItem(PPLibItem('recogFive', recogDigitType, None))

def addRecogFive2(lib):
    inType = mkRealTensorSort([1, 1, 28, 28])
    outType = mkBoolTensorSort([1, 1])
    recogDigitType = mkFuncSort(inType, outType)
    lib.addItem(PPLibItem('recogFive2', recogDigitType, None))

def getCountFiveSort():
    inType = mkListSort(mkRealTensorSort([1, 1, 28, 28]))
    outType = mkRealTensorSort([1, 1])
    return mkFuncSort(inType, outType)

def getCountFiveProg():
    return PPFuncApp(fn=PPVar(name='lib.compose'),
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

def getCountFiveLib():
    lib = mkDefaultLib()
    addRecogFive(lib)
    return lib

def testMutate0():
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

    for i in range(100):
        newProg = mutate(prog, lib)
        if newProg is None:
            continue
        print(repr_py(newProg))

        isort = inferType(newProg, lib)
        assert isort is not None
        assert sort == isort


def testMutate1():
    lib = mkDefaultLib()

    addRecogFive(lib)

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

    for i in range(100):
        newProg = mutate(prog, lib)
        if newProg is None:
            continue
        # print(repr_py(newProg))
        # print('newProg: ', newProg)
        # print('newProgReprPy: ', repr_py(newProg))
        isort = inferType(newProg, lib)
        if isort is None:
            print(repr_py(newProg))
            print('newProg: ', newProg)
            continue

        assert sort == isort


def testMutate2():
    """
    Unkprog
    """
    lib = mkDefaultLib()

    addRecogFive(lib)

    inType = mkListSort(mkRealTensorSort([1, 1, 28, 28]))
    outType = mkRealTensorSort([1, 1])
    sort = mkFuncSort(inType, outType)

    prog = \
        PPTermUnk(name='nn_fun_x_906', sort=sort)

    for i in range(100):
        newProg = mutate(prog, lib)
        if newProg is None:
            continue
        # print(repr_py(newProg))
        # print('newProg: ', newProg)
        # print('newProgReprPy: ', repr_py(newProg))
        isort = inferType(newProg, lib)
        if isort is None:
            print(repr_py(newProg))
            print('newProg: ', newProg)
            continue

        assert sort == isort


def testProgGenerator():
    lib = mkDefaultLib()

    pg = ProgramGenerator(lib)

    # Tensor[real][1,1]
    sort = PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)])

    for i in range(100):
        prog = pg.genProg(sort)
        print(repr_py_ann(prog))


def testProgGenerator2():
    lib = getCountFiveLib()

    pg = ProgramGenerator(lib)

    sort = getCountFiveSort()

    for i in range(1000):
        prog = pg.genProg(sort)
        if prog is not None:
            print(repr_py(prog))


def testAvoidDimConstUnk():
    term = PPFuncApp(fn=PPVar(name='lib.zeros'), args=[PPTermNT(name='Z', sort=PPDimConst(value=1))])
    ntId = 1
    res = expandToUnk(term, ntId)
    assert res is None


def testIdTermSortMap():
    lib = mkDefaultLib()
    addRecogFive(lib)

    def getSort():
        inType = mkListSort(mkRealTensorSort([1, 1, 28, 28]))
        outType = mkRealTensorSort([1, 1])
        return mkFuncSort(inType, outType)
    sort = getSort()

    pg = ProgramGenerator(lib)
    bad = 0
    for i in range(100):
        prog = pg.genProg(sort)
        if prog is None:
            continue

        try:
            m = getIdTermSorts(prog, lib)
        except Exception as e:
            bad += 1
            print('#######')
            print(prog)
            print('')

        print(prog)
        print(m)
    print("BadProgram: " + str(bad))


def testIdTermSortMap2():
    lib = mkDefaultLib()
    addRecogFive(lib)

    prog = PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPTermUnk(name='Unk', sort=PPFuncSort(args=[PPFuncSort(args=[
        PPListSort(param_sort=PPListSort(param_sort=PPSortVar(name='A_1_2')))], rtpe=PPListSort(param_sort=PPListSort(param_sort=PPSortVar(name='B'))))], rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))), PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPVar(name='lib.map_l'), PPVar(name='lib.map_l')]), PPTermUnk(name='Unk', sort=PPFuncSort(args=[PPListSort(param_sort=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1), PPDimConst(value=28), PPDimConst(value=28)]))], rtpe=PPFuncSort(args=[PPSortVar(name='A_1_2')], rtpe=PPSortVar(name='B'))))])])

    m = getIdTermSorts(prog, lib)
    m2 = {0: (PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPTermUnk(name='Unk', sort=PPFuncSort(args=[PPFuncSort(
        args=[
        PPListSort(param_sort=PPListSort(param_sort=PPSortVar(name='A_1_2')))], rtpe=PPListSort(param_sort=PPListSort(param_sort=PPSortVar(name='B'))))], rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))), PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPVar(name='lib.map_l'), PPVar(name='lib.map_l')]), PPTermUnk(name='Unk', sort=PPFuncSort(args=[PPListSort(param_sort=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1), PPDimConst(value=28), PPDimConst(value=28)]))], rtpe=PPFuncSort(args=[PPSortVar(name='A_1_2')], rtpe=PPSortVar(name='B'))))])]), None), 1: (PPTermUnk(name='Unk', sort=PPFuncSort(args=[PPFuncSort(args=[PPListSort(param_sort=PPListSort(param_sort=PPSortVar(name='A_1_2')))], rtpe=PPListSort(param_sort=PPListSort(param_sort=PPSortVar(name='B'))))], rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))), PPFuncSort(args=[PPFuncSort(args=[PPListSort(param_sort=PPListSort(param_sort=PPSortVar(name='A_1_2')))], rtpe=PPListSort(param_sort=PPListSort(param_sort=PPSortVar(name='B'))))], rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))), 2: (PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPVar(name='lib.map_l'), PPVar(name='lib.map_l')]), PPTermUnk(name='Unk', sort=PPFuncSort(args=[PPListSort(param_sort=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1), PPDimConst(value=28), PPDimConst(value=28)]))], rtpe=PPFuncSort(args=[PPSortVar(name='A_1_2')], rtpe=PPSortVar(name='B'))))]), None), 3: (PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPVar(name='lib.map_l'), PPVar(name='lib.map_l')]), None), 4: (PPVar(name='lib.map_l'), PPFuncSort(args=[PPFuncSort(args=[PPSortVar(name='A')], rtpe=PPSortVar(name='B'))], rtpe=PPFuncSort(args=[PPListSort(param_sort=PPSortVar(name='A'))], rtpe=PPListSort(param_sort=PPSortVar(name='B'))))), 5: (PPVar(name='lib.map_l'), PPFuncSort(args=[PPFuncSort(args=[PPSortVar(name='A')], rtpe=PPSortVar(name='B'))], rtpe=PPFuncSort(args=[PPListSort(param_sort=PPSortVar(name='A'))], rtpe=PPListSort(param_sort=PPSortVar(name='B'))))), 6: (PPTermUnk(name='Unk', sort=PPFuncSort(args=[PPListSort(param_sort=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1), PPDimConst(value=28), PPDimConst(value=28)]))], rtpe=PPFuncSort(args=[PPSortVar(name='A_1_2')], rtpe=PPSortVar(name='B')))), PPFuncSort(args=[PPListSort(param_sort=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1), PPDimConst(value=28), PPDimConst(value=28)]))], rtpe=PPFuncSort(args=[PPSortVar(name='A_1_2')], rtpe=PPSortVar(name='B'))))}

    assert m == m2


def testCrossover():
    def getSort():
        inType = mkListSort(mkRealTensorSort([1, 1, 28, 28]))
        outType = mkRealTensorSort([1, 1])
        return mkFuncSort(inType, outType)

    def genProgs(aSort, aLib):
        pg = ProgramGenerator(aLib)
        for i in range(100):
            prog = pg.genProg(aSort)
            if prog is None:
                continue
            yield prog

    lib = mkDefaultLib()
    addRecogFive(lib)
    addRecogFive2(lib)

    sort = getSort()
    progs = list(genProgs(sort, lib))

    i = 0
    half = len(progs)//2
    print('half: ', half)
    for i in range(half):
        p1, p2 = progs[i], progs[i + half]
        c1, c2 = crossover(p1, p2, lib)
        if p1 == c1 and p2 == c2:
            i += 1
            print('p1')
            print(p1)
            print('p2')
            print(p2)
        else:
            print('p1')
            print(p1)
            print('p2')
            print(p2)
            print('c1')
            print(c1)
            print('c2')
            print(c2)
            print('')

    print('failed crossovers: %d' % i)


def testCrossover2():
    lib = mkDefaultLib()
    addRecogFive(lib)
    p1 = PPTermUnk(name='Unk', sort=PPFuncSort(args=[PPListSort(param_sort=PPTensorSort(param_sort=PPReal(),
                                                                                   shape=[PPDimConst(value=1),
                                                                                          PPDimConst(value=1),
                                                                                          PPDimConst(value=28),
                                                                                          PPDimConst(value=28)]))],
                                          rtpe=PPTensorSort(param_sort=PPReal(),
                                                            shape=[PPDimConst(value=1), PPDimConst(value=1)])))
    p2 = PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPTermUnk(name='Unk',
                                                                                                          sort=PPFuncSort(
                                                                                                              args=[
                                                                                                                  PPSortVar(
                                                                                                                      name='B_1')],
                                                                                                              rtpe=PPTensorSort(
                                                                                                                  param_sort=PPReal(),
                                                                                                                  shape=[
                                                                                                                      PPDimConst(
                                                                                                                          value=1),
                                                                                                                      PPDimConst(
                                                                                                                          value=1)]))),
                                                                                                PPTermUnk(name='Unk',
                                                                                                          sort=PPFuncSort(
                                                                                                              args=[
                                                                                                                  PPSortVar(
                                                                                                                      name='C')],
                                                                                                              rtpe=PPSortVar(
                                                                                                                  name='B_1')))]),
                                                  PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPTermUnk(name='Unk',
                                                                                                          sort=PPFuncSort(
                                                                                                              args=[
                                                                                                                  PPSortVar(
                                                                                                                      name='B')],
                                                                                                              rtpe=PPSortVar(
                                                                                                                  name='C'))),
                                                                                                PPTermUnk(name='Unk',
                                                                                                          sort=PPFuncSort(
                                                                                                              args=[
                                                                                                                  PPListSort(
                                                                                                                      param_sort=PPTensorSort(
                                                                                                                          param_sort=PPReal(),
                                                                                                                          shape=[
                                                                                                                              PPDimConst(
                                                                                                                                  value=1),
                                                                                                                              PPDimConst(
                                                                                                                                  value=1),
                                                                                                                              PPDimConst(
                                                                                                                                  value=28),
                                                                                                                              PPDimConst(
                                                                                                                                  value=28)]))],
                                                                                                              rtpe=PPSortVar(
                                                                                                                  name='B')))])])
    c1, c2 = crossover(p1, p2, lib)
    assert p1 == c1
    assert p2 == c2


def testCrossover3():
    lib = mkDefaultLib()
    addRecogFive(lib)
    addRecogFive2(lib)

    countFive1 = \
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


    countFive2 = \
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
                        PPFuncApp(fn=PPVar(name='lib.map_l'), args=[PPVar(name='lib.recogFive2')])])

    c1, c2 = crossover(countFive1, countFive2, lib)
    assert c1 == countFive2
    assert c2 == countFive1


def testSymbolicSynthesizerEA():
    lib = getCountFiveLib()
    sort = getCountFiveSort()
    synth = SymbolicSynthesizerEA(lib, sort)
    scores = None
    for gen in range(2):
        progs = synth.genProgs(scores)
        print('Generation %d ############' % gen)
        for prog, _ in progs:
            # print(prog)
            print(repr_py(prog))
        scores = [random.uniform(0, 1.0) for _ in progs]
