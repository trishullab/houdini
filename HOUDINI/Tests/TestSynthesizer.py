import torch.nn.functional as F
from HOUDINI.Interpreter.NeuralModules import NetCNN
from HOUDINI.NeuralSynthesizer import NeuralSynthesizer

from Data import split_into_train_and_validation, get_batch_count_iseven
from HOUDINI.Interpreter.Interpreter import Interpreter
from HOUDINI.FnLibraryFunctions import pp_map, pp_reduce, get_items_from_repo
from HOUDINI.FnLibrary import FnLibrary, PPLibItem
from HOUDINI.Synthesizer import ASTUtils
from HOUDINI.Synthesizer.AST import *
from HOUDINI.Synthesizer.ASTDSL import mkTensorSort, mkFuncSort, mkListSort, mkRealTensorSort, \
    mkBoolTensorSort, mkIntTensorSort, mkGraphSort
from HOUDINI.Synthesizer.ReprUtils import repr_py_ann, repr_py
from HOUDINI.Synthesizer.SymbolicSynthesizer import SymbolicSynthesizer


def test1():
    intSort = PPInt()
    boolSort = PPBool()

    libSynth = FnLibrary()
    libSynth.addItems([PPLibItem('itob', mkFuncSort(intSort, boolSort), None), ])
    ioExamples = None

    fnSort = PPFuncSort([intSort], boolSort)
    interpreter = None

    solver = SymbolicSynthesizer(interpreter, libSynth, fnSort, ioExamples)
    solver.setEvaluate(False)
    targetProg = PPVar('lib.itobX')
    count = solver.search(targetProg, 100)
    assert count == -1


def xtest2():
    intSort = PPInt()
    boolSort = PPBool()
    libSynth = FnLibrary()
    libSynth.addItems([PPLibItem('itob', mkFuncSort(intSort, boolSort), None), ])
    ioExamples = None

    fnSort = PPFuncSort([intSort], boolSort)
    interpreter = None

    solver = SymbolicSynthesizer(interpreter, libSynth, fnSort, ioExamples)
    solver.setEvaluate(False)
    targetProg = PPLambda(params=[PPVarDecl(name='x1', sort=PPInt())],
                          body=PPFuncApp(fn=PPVar('lib.itob'), args=[PPVar(name='x1')]))
    count = solver.search(targetProg, 100)
    assert count >= 0


def test3():
    t123 = mkTensorSort(PPInt(), [1, 2, 3])
    t12 = mkTensorSort(PPInt(), [1, 2])
    libSynth = FnLibrary()
    libSynth.addItems([PPLibItem('one', mkFuncSort(t123, t12), None), ])
    ioExamples = None

    fnSort = PPFuncSort([t123], t12)
    interpreter = None

    solver = SymbolicSynthesizer(interpreter, libSynth, fnSort, ioExamples)
    solver.setEvaluate(False)
    solution, score = solver.solve()
    print(solution)
    print(score)


def xtest4():
    t123 = mkTensorSort(PPInt(), [1, 2, 3])
    t333 = mkTensorSort(PPInt(), [3, 3, 3])
    tabc = mkTensorSort(PPInt(), ['a', 'b', 'c'])
    tabcc = mkTensorSort(PPInt(), ['a', 'b', 'c', 'd'])
    tabcd = mkTensorSort(PPInt(), ['a', 'b', 'c', 'd'])
    tddd = mkTensorSort(PPInt(), ['d', 'd', 'd'])

    libSynth = FnLibrary()
    libSynth.addItems([PPLibItem('one', mkFuncSort(tabc, tabcc), None), ])
    libSynth.addItems([PPLibItem('two', mkFuncSort(tabcd, tddd), None), ])

    ioExamples = None

    fnSort = PPFuncSort([t123], t333)

    interpreter = None

    solver = SymbolicSynthesizer(interpreter, libSynth, fnSort, ioExamples)
    solver.setEvaluate(False)
    solution, score = solver.solve()
    print(solution)
    print(score)


def test5():
    def mk_recognise_5s():
        res = NetCNN("recognise_5s", input_ch=1, output_dim=1, output_activation=F.sigmoid)
        res.load('../Interpreter/Models/is5_classifier.pth.tar')
        return res

    libSynth = FnLibrary()

    t = PPSortVar('T')
    t1 = PPSortVar('T1')
    t2 = PPSortVar('T2')

    libSynth.addItems([
        PPLibItem('recognise_5s', mkFuncSort(mkTensorSort(PPReal(), ['a', 1, 28, 28]),
                                             mkTensorSort(PPReal(), ['a', 1])), mk_recognise_5s()),
        PPLibItem('map', mkFuncSort(mkFuncSort(t1, t2), mkListSort(t1), mkListSort(t2)), pp_map),
    ])

    ioExamples = None
    img = mkRealTensorSort([1, 1, 28, 28])
    imgList = mkListSort(img)
    isFive = mkRealTensorSort([1, 1])
    imgToIsFive = mkFuncSort(img, isFive)
    isFiveList = mkListSort(isFive)

    fnSort = mkFuncSort(imgList, isFiveList)

    interpreter = None

    """targetProg = lambda inputs: map(lib.recognise_5s, inputs)"""

    solver = SymbolicSynthesizer(interpreter, libSynth, fnSort, ioExamples, ioExamples)
    solver.setEvaluate(False)
    # TODO: use "search" instead of "solve"
    solution, score = solver.solve()
    print(solution)
    print(score)


def test6():
    t = PPSortVar('T')
    t1 = PPSortVar('T1')
    t2 = PPSortVar('T2')

    def mk_recognise_5s():
        res = NetCNN("recognise_5s", input_ch=1, output_dim=1, output_activation=F.sigmoid)
        res.load('../Interpreter/Models/is5_classifier.pth.tar')
        return res

    libSynth = FnLibrary()

    real_tensor_2d = mkTensorSort(PPReal(), ['a', 'b'])
    libSynth.addItems([
        PPLibItem('recognise_5s', mkFuncSort(mkTensorSort(PPReal(), ['a', 1, 28, 28]),
                                             mkTensorSort(PPReal(), ['a', 1])), mk_recognise_5s()),
        PPLibItem('map', mkFuncSort(mkFuncSort(t1, t2), mkListSort(t1), mkListSort(t2)), pp_map),
        PPLibItem('reduce', mkFuncSort(mkFuncSort(t, t, t), mkListSort(t), t), pp_reduce),
        PPLibItem('add', mkFuncSort(real_tensor_2d, real_tensor_2d, real_tensor_2d), lambda x, y: x + y),
    ])

    train, val = split_into_train_and_validation(0, 10)
    val_ioExamples = get_batch_count_iseven(digits_to_count=[5], count_up_to=10, batch_size=20, digit_dictionary=val)

    img = mkRealTensorSort([1, 1, 28, 28])
    isFive = mkRealTensorSort([1, 1])
    imgToIsFive = mkFuncSort(img, isFive)
    imgList = mkListSort(img)
    isFiveList = mkListSort(isFive)

    sumOfFives = mkRealTensorSort([1, 1])

    fnSort = mkFuncSort(imgList, sumOfFives)

    interpreter = Interpreter(libSynth)
    """
    targetProg = 
        lambda inputs. 
            reduce( 
                add, 
                map(lib.recognise_5s, inputs))
    """
    # TODO: use "search" instead of "solve"
    solver = SymbolicSynthesizer(interpreter, libSynth, fnSort, val_ioExamples, val_ioExamples)
    # solver.setEvaluate(False)
    solution, score = solver.solve()
    print(solution)
    print(score)


def test_zeros():
    # IO Examples
    train, val = split_into_train_and_validation(0, 10)
    train_io_examples = get_batch_count_iseven(digits_to_count=[5], count_up_to=10, batch_size=100,
                                               digit_dictionary=train)
    val_io_examples = get_batch_count_iseven(digits_to_count=[5], count_up_to=10, batch_size=20, digit_dictionary=val)

    def mk_recognise_5s():
        res = NetCNN("recognise_5s", input_ch=1, output_dim=1, output_activation=F.sigmoid)
        res.load('../Interpreter/Models/is5_classifier.pth.tar')
        return res

    # Library
    libSynth = FnLibrary()

    t = PPSortVar('T')
    t1 = PPSortVar('T1')
    t2 = PPSortVar('T2')

    libSynth.addItems([
        PPLibItem('zeros', mkFuncSort(PPDimVar('a'), mkRealTensorSort([1, 'a'])), pp_map),
        # PPLibItem('zeros2', mkFuncSort(PPDimVar('a'), PPDimVar('b'), mkRealTensorSort(['a', 'b'])), pp_map),
    ])

    fnSort = mkFuncSort(PPDimConst(2), mkRealTensorSort([2]))

    interpreter = Interpreter(libSynth)
    solver = SymbolicSynthesizer(interpreter, libSynth, fnSort, train_io_examples, val_io_examples)
    solver.setEvaluate(False)
    solution, score = solver.solve()


def func(*lst):
    return mkFuncSort(*lst)


def lst(t):
    return mkListSort(t)


def graph(t):
    return mkGraphSort(t)


def getLib():
    libSynth = FnLibrary()
    A = PPSortVar('A')
    B = PPSortVar('B')
    C = PPSortVar('C')

    tr5 = mkRealTensorSort([5])
    tb5 = mkBoolTensorSort([5])
    ti5 = mkIntTensorSort([5])
    ppint = PPInt()

    cnts = PPEnumSort(2, 50)

    libSynth.addItems([
        PPLibItem('map', func(func(A, B), func(lst(A), lst(B))), None),
        PPLibItem('fold', func(func(B, A, B), B, func(lst(A), B)), None),
        PPLibItem('conv', func(func(A, lst(A), A), func(lst(A), lst(A))), None),
        PPLibItem('compose', func(func(B, C), func(A, B), func(A, C)), None),
        PPLibItem('repeat', func(cnts, func(A, A), func(A, A)), None),
        PPLibItem('zeros', func(PPDimVar('a'), mkRealTensorSort([1, 'a'])), None),
    ])
    return libSynth


def test_synthesizer_count_5s():
    libSynth = getLib()

    tr1_1_28_28 = mkRealTensorSort([1, 1, 28, 28])
    tr1_1 = mkRealTensorSort([1, 1])
    tb1_1 = mkBoolTensorSort([1, 1])

    libSynth.addItems([PPLibItem('recog_5', func(tr1_1_28_28, tb1_1), None), ])

    fn_sort = func(lst(tr1_1_28_28), tr1_1)

    synth = SymbolicSynthesizer(libSynth, fn_sort)

    I = 10000
    i = 0
    for prog, unkMap in synth.genProgs():

        i = i + 1
        if i > I:
            break
        if i % 100 == 0:
            print(i)

        unks = ASTUtils.getUnks(prog)

        if len(unks) > 1:
            continue

        for unk in unks:
            if ASTUtils.isAbstract(unk.sort):
                continue

        print(repr_py_ann(prog))


def test_synthesizer_sum_digits():
    libSynth = getLib()

    input_type = mkListSort(mkRealTensorSort([1, 1, 28, 28]))
    output_type = mkRealTensorSort([1, 1])
    fn_sort = mkFuncSort(input_type, output_type)

    tr1_1_28_28 = mkRealTensorSort([1, 1, 28, 28])
    tb_1_10 = mkBoolTensorSort([1, 10])
    classify_digit = mkFuncSort(tr1_1_28_28, tb_1_10)

    libSynth.addItems([PPLibItem('classify_digit', classify_digit, None), ])

    synth = SymbolicSynthesizer(libSynth, fn_sort)

    I = 10000
    i = 0
    for prog, unkMap in synth.genProgs():

        i = i + 1
        if i > I:
            break
        if i % 100 == 0:
            print(i)

        unks = ASTUtils.getUnks(prog)

        if len(unks) > 1:
            continue

        for unk in unks:
            if ASTUtils.isAbstract(unk.sort):
                continue

        print(repr_py_ann(prog))


def get_synth_lib():
    libSynth = FnLibrary()
    A = PPSortVar('A')
    B = PPSortVar('B')
    C = PPSortVar('C')

    tr5 = mkRealTensorSort([5])
    tb5 = mkBoolTensorSort([5])
    ti5 = mkIntTensorSort([5])
    ppint = PPInt()

    repeatEnum = PPEnumSort(10, 10)

    libSynth.addItems([
        PPLibItem('compose', func(func(B, C), func(A, B), func(A, C)), None),
        PPLibItem('map_l', func(func(A, B), func(lst(A), lst(B))), None),
        PPLibItem('fold_l', func(func(B, A, B), B, func(lst(A), B)), None),
        PPLibItem('conv_l', func(func(lst(A), B), func(lst(A), lst(B))), None),
        PPLibItem('conv_g', func(func(lst(A), B), func(graph(A), graph(B))), None),
        PPLibItem('map_g', func(func(A, B), func(graph(A), graph(B))), None),
        PPLibItem('fold_g', func(func(B, A, B), B, func(graph(A), B)), None),
        PPLibItem('zeros', func(PPDimVar('a'), mkRealTensorSort([1, 'a'])), None),
        PPLibItem('repeat', func(repeatEnum, func(A, A), func(A, A)), None),
        PPLibItem('regress_speed_mnist', func(mkRealTensorSort([1, 3, 32, 32]), mkRealTensorSort([1, 2])), None),

        # PPLibItem('nav_mnist', func(mkGraphSort(mkRealTensorSort([1, 3, 32, 32])),
        #                             mkGraphSort(mkRealTensorSort([1, 2]))), None),

    ])

    return libSynth


def test_synthesizer_graph():
    libSynth = get_synth_lib()

    input_type = mkGraphSort(mkRealTensorSort([1, 3, 32, 32]))
    output_type = mkGraphSort(mkRealTensorSort([1, 2]))
    fn_sort = mkFuncSort(input_type, output_type)

    synth = SymbolicSynthesizer(libSynth, fn_sort)

    I = 20
    i = 0
    for prog, unkMap in synth.genProgs():
        if i > I:
            break

        if NeuralSynthesizer.is_evaluable(prog)[0]:
            i = i + 1
            # print(i, repr_py_ann(prog))
            print(i, repr_py(prog))


            # compose.*repeat.*conv_g.*map_g
            # compose( repeat(N, convg(Z),  mapg(regress_speed))


def getBaseLibrary():
    libSynth = FnLibrary()

    libSynth.addItems(get_items_from_repo(['compose',
                                           'map_l', 'fold_l', 'conv_l',
                                           'conv_g', 'map_g', 'fold_g',
                                           'zeros', 'repeat'
                                           ]))
    return libSynth


def progSize(prog):
    return 0


def test_search_space():
    # type shortcuts
    timg = mkRealTensorSort([1, 1, 28, 28])
    treal = mkRealTensorSort([1, 1])
    tbool = mkBoolTensorSort([1, 1])
    tbool10 = mkBoolTensorSort([1, 10])
    treal2 = mkRealTensorSort([1, 2])

    def mklst(t):
        return mkListSort(t)

    def mkfn(t1, t2):
        return mkFuncSort(t1, t2)

    def mkgr(t):
        return mkGraphSort(t)

    targetSize = 6

    print('start')
    print('targetSize = %d' % targetSize)
    add_sort = mkFuncSort(treal, tbool, treal)

    testSeqs = ['cs1']

    if 'cs1' in testSeqs:
        sname = 'cs1'
        libSynth = getBaseLibrary()

        tname = sname + '_recog_digit_d1'
        print(tname)
        fn_sort = mkfn(timg, tbool)
        print_progs(fn_sort, libSynth, targetSize)
        lfn = fn_sort
        add_to_lib(lfn, libSynth, tname)

        tname = sname + '_recog_digit_d2'
        print(tname)
        fn_sort = mkfn(timg, tbool)
        print_progs(fn_sort, libSynth, targetSize)
        lfn = fn_sort
        add_to_lib(lfn, libSynth, tname)

        tname = sname + '_count_digit_d1'
        print(tname)
        fn_sort = mkfn(mklst(timg), treal)
        print_progs(fn_sort, libSynth, targetSize)
        lfn = add_sort
        add_to_lib(lfn, libSynth, 'nn_add')

        tname = sname + '_count_digit_d2'
        print(tname)
        fn_sort = mkfn(mklst(timg), treal)
        print_progs(fn_sort, libSynth, targetSize)

    if 'cs2' in testSeqs:
        sname = 'cs2'
        libSynth = getBaseLibrary()

        tname = sname + '_recog_digit_d1'
        print(tname)
        fn_sort = mkfn(timg, tbool)
        print_progs(fn_sort, libSynth, targetSize)
        lfn = fn_sort
        add_to_lib(lfn, libSynth, tname)

        tname = sname + '_count_digit_d1'
        print(tname)
        fn_sort = mkfn(mklst(timg), treal)
        print_progs(fn_sort, libSynth, targetSize)
        lfn = add_sort
        add_to_lib(fn_sort, libSynth, 'nn_add')

        tname = sname + '_count_digit_d2'
        print(tname)
        fn_sort = mkfn(mklst(timg), treal)
        print_progs(fn_sort, libSynth, targetSize)
        lfn = mkfn(timg, tbool)
        add_to_lib(fn_sort, libSynth, 'nn_recog_digit_d2')

        tname = sname + '_recog_digit_d2'
        print(tname)
        fn_sort = mkfn(timg, tbool)
        print_progs(fn_sort, libSynth, targetSize)

    if 'ss' in testSeqs:
        sname = 'ss'
        libSynth = getBaseLibrary()

        tname = sname + '_classify_digit'
        print(tname)
        fn_sort = mkfn(timg, tbool10)
        print_progs(fn_sort, libSynth, targetSize)
        lfn = fn_sort
        add_to_lib(fn_sort, libSynth, tname)

        tname = sname + '_sum_digits'
        print(tname)
        fn_sort = mkfn(mklst(timg), treal)
        print_progs(fn_sort, libSynth, targetSize)

    if 'gs1' in testSeqs:
        sname = 'gs1'
        libSynth = getBaseLibrary()

        tname = sname + '_regress_speed'
        print(tname)
        fn_sort = mkfn(timg, treal2)
        print_progs(fn_sort, libSynth, targetSize)
        lfn = fn_sort
        add_to_lib(fn_sort, libSynth, tname)

        tname = sname + '_shortest_path_street'
        print(tname)
        fn_sort = mkfn(mkgr(timg), mkgr(treal2))
        print_progs(fn_sort, libSynth, targetSize)

    if 'gs2' in testSeqs:
        sname = 'gs2'
        libSynth = getBaseLibrary()

        tname = sname + '_regress_mnist'
        print(tname)
        fn_sort = mkfn(timg, treal2)
        # print_progs(fn_sort, libSynth, targetSize)
        lfn = fn_sort
        add_to_lib(lfn, libSynth, tname)

        tname = sname + '_shortest_path_mnist'
        print(tname)
        fn_sort = mkfn(mkgr(timg), mkgr(treal2))
        # print_progs(fn_sort, libSynth, targetSize)
        lfn = mkfn(mklst(treal2), treal2)
        add_to_lib(lfn, libSynth, 'nn_relax')

        tname = sname + '_shortest_path_street'
        print(tname)
        fn_sort = mkfn(mkgr(timg), mkgr(treal2))
        print_progs(fn_sort, libSynth, targetSize)


def add_to_lib(fn_sort, libSynth, tname):
    libSynth.addItem(PPLibItem(tname, fn_sort, None))


def print_progs(fn_sort, libSynth, targetSize):
    synth = SymbolicSynthesizer(libSynth, fn_sort)
    cnt = 0
    for prog, unkMap in synth.genProgs():
        sz = ASTUtils.getSize2(prog)
        if sz > targetSize:
            break

        cnt += 1
        print(cnt, sz, repr_py(prog))
        # print(cnt, sz, prog)

    print(cnt)


def print_progs_e(fn_sort, libSynth, targetSize):
    synth = SymbolicSynthesizer(libSynth, fn_sort)
    cnt = 1
    ecnt = 1
    for prog, unkMap in synth.genProgs():
        sz = ASTUtils.getSize2(prog)
        if sz > targetSize:
            break

        res, code = NeuralSynthesizer.is_evaluable(prog, fn_sort)
        if res:
            if 'lib.compose(lib.repeat(10, lib.conv_g(lib.nn_relax))' in repr_py(prog):
                print('.', cnt, sz, repr_py(prog), ecnt)
            ecnt += 1
        else:
            if 'lib.compose(lib.repeat(10, lib.conv_g(lib.nn_relax))' in repr_py(prog):
                print("#", cnt, sz, repr_py(prog), code)
                test = False
                if test:
                    res, code = NeuralSynthesizer.is_evaluable(prog, fn_sort)
        # print(cnt, sz, prog)
        cnt += 1
    print(cnt)
    # lib.compose(lib.repeat(10, lib.conv_g(lib.nn_relax)),
