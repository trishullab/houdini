from HOUDINI.FnLibrary import FnLibrary, PPLibItem
from HOUDINI.Synthesizer import ReprUtils, Rules
from HOUDINI.Synthesizer.AST import *
from HOUDINI.Synthesizer.ASTDSL import mkFuncSort, mkListSort, mkRealTensorSort, \
    mkBoolTensorSort, mkIntTensorSort
from HOUDINI.Synthesizer.ReprUtils import *


def printTerm(t):
    # print('++++++++++++++')
    # print(ReprUtils.repr_py_ann(t))
    # print('')
    printCodeGen(t)


def printCodeGen(t):
    print('assert(t == %s)' % str(t))
    print('printTerm(t)')
    print('#%s' % ReprUtils.repr_py_ann(t))
    print('')


def func(*lst):
    return mkFuncSort(*lst)


def lst(t):
    return mkListSort(t)


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
        PPLibItem('nn_fun_0', func(tr5, tr5), None),
        PPLibItem('nn_fun_1', func(tr5, tb5), None),
        PPLibItem('nn_fun_2', func(tb5, ti5), None),
    ])
    return libSynth


tr5 = mkRealTensorSort([5])
tb5 = mkBoolTensorSort([5])
ti5 = mkIntTensorSort([5])


def test_repeat():
    libSynth = getLib()
    t = PPTermNT("Z", func(tr5, tr5))
    assert (t == PPTermNT(name='Z',
                          sort=PPFuncSort(args=[PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=5)])],
                                          rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=5)]))))
    printTerm(t)

    t = Rules.expandToFuncApp(libSynth, t, 1, 'repeat')
    assert (t == PPFuncApp(fn=PPVar(name='lib.repeat'), args=[PPTermNT(name='Z', sort=PPEnumSort(start=2, end=50)),
                                                              PPTermNT(name='Z', sort=PPFuncSort(args=[
                                                                  PPTensorSort(param_sort=PPReal(),
                                                                               shape=[PPDimConst(value=5)])],
                                                                  rtpe=PPTensorSort(
                                                                      param_sort=PPReal(),
                                                                      shape=[PPDimConst(
                                                                          value=5)])))]));
    printTerm(t)

    t = Rules.expandEnum(t, 1, PPIntConst(5))
    assert (t == PPFuncApp(fn=PPVar(name='lib.repeat'), args=[PPIntConst(value=5), PPTermNT(name='Z', sort=PPFuncSort(
        args=[PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=5)])],
        rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=5)])))]));  #

    printTerm(t)


def test_conv():
    libSynth = getLib()
    t = PPTermNT("Z", func(lst(tr5), lst(tr5)))
    printTerm(t)

    t = Rules.expandToFuncApp(libSynth, t, 1, 'conv')
    assert (t == PPFuncApp(fn=PPVar(name='lib.conv'), args=[PPTermNT(name='Z', sort=PPFuncSort(
        args=[PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=5)]),
              PPListSort(param_sort=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=5)]))],
        rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=5)])))]))
    printTerm(t)

    t = Rules.expandToUnk(t, 1)
    assert (t == PPFuncApp(fn=PPVar(name='lib.conv'), args=[PPTermUnk(name='Unk', sort=PPFuncSort(
        args=[PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=5)]),
              PPListSort(param_sort=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=5)]))],
        rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=5)])))]))
    printTerm(t)


def test_map():
    libSynth = getLib()
    # map
    t = PPTermNT("Z", func(lst(tr5), lst(tr5)))
    printTerm(t)
    t = Rules.expandToFuncApp(libSynth, t, 1, 'map')
    assert (t == PPFuncApp(fn=PPVar(name='lib.map'), args=[PPTermNT(name='Z', sort=PPFuncSort(
        args=[PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=5)])],
        rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=5)])))]))

    printTerm(t)

    t = Rules.expandToVar(libSynth, t, 1, 'nn_fun_0')
    assert (t == PPFuncApp(fn=PPVar(name='lib.map'), args=[PPVar(name='lib.nn_fun_0')]))
    printTerm(t)


def test_fold():
    tr15 = mkRealTensorSort([1, 5])
    tb51 = mkBoolTensorSort([1, 5])
    ti15 = mkIntTensorSort([1, 5])

    libSynth = getLib()

    t = PPTermNT("Z", func(lst(tr15), tr15))
    printTerm(t)
    # (Z: (List[Tensor[real][1,5]] --> Tensor[real][1,5]))
    t = Rules.expandToFuncApp(libSynth, t, 1, 'fold')
    assert (t == PPFuncApp(fn=PPVar(name='lib.fold'), args=[PPTermNT(name='Z', sort=PPFuncSort(
        args=[PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=5)]),
              PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=5)])],
        rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=5)]))),
                                                            PPTermNT(name='Z',
                                                                     sort=PPTensorSort(
                                                                         param_sort=PPReal(),
                                                                         shape=[
                                                                             PPDimConst(
                                                                                 value=1),
                                                                             PPDimConst(
                                                                                 value=5)]))])
            )

    printTerm(t)
    # lib.fold((Z: ((Tensor[real][1,5], Tensor[real][1,5]) --> Tensor[real][1,5])), (Z: Tensor[real][1,5]))
    t = Rules.expandToFuncApp(libSynth, t, 2, 'zeros')

    assert (t == PPFuncApp(fn=PPVar(name='lib.fold'), args=[PPTermNT(name='Z', sort=PPFuncSort(
        args=[PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=5)]),
              PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=5)])],
        rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=5)]))),
                                                            PPFuncApp(fn=PPVar(name='lib.zeros'), args=[
                                                                PPTermNT(name='Z', sort=PPDimConst(value=5))])])
            )

    printTerm(t)
    # lib.fold((Z: ((Tensor[real][1,5], Tensor[real][1,5]) --> Tensor[real][1,5])), lib.zeros((Z: 5)))
    t = Rules.expandDimConst(t, 2)
    assert (t == PPFuncApp(fn=PPVar(name='lib.fold'), args=[PPTermNT(name='Z', sort=PPFuncSort(
        args=[PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=5)]),
              PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=5)])],
        rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=5)]))),
                                                            PPFuncApp(fn=PPVar(name='lib.zeros'),
                                                                      args=[PPIntConst(value=5)])])
            )

    printTerm(t)
    # lib.fold((Z: ((Tensor[real][1,5], Tensor[real][1,5]) --> Tensor[real][1,5])), lib.zeros(5))
    t = Rules.expandToUnk(t, 1)
    assert (t == PPFuncApp(fn=PPVar(name='lib.fold'), args=[PPTermUnk(name='Unk', sort=PPFuncSort(
        args=[PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=5)]),
              PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=5)])],
        rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=5)]))),
                                                            PPFuncApp(fn=PPVar(name='lib.zeros'),
                                                                      args=[PPIntConst(value=5)])]))

    printTerm(t)
    # lib.fold((Unk: ((Tensor[real][1,5], Tensor[real][1,5]) --> Tensor[real][1,5])), lib.zeros(5))


def test_unk():
    libSynth = getLib()
    t = PPTermNT("Z", func(lst(tr5), tr5))
    printTerm(t)
    t = Rules.expandToUnk(t, 1)
    assert (t == PPTermUnk(name='Unk', sort=PPFuncSort(
        args=[PPListSort(param_sort=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=5)]))],
        rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=5)]))))
    printTerm(t)


def test_compose():
    libSynth = getLib()
    t = PPTermNT("Z", func(tr5, ti5))
    printTerm(t)
    t = Rules.expandToFuncApp(libSynth, t, 1, 'compose')
    assert (t == PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPTermNT(name='Z',
                                                                        sort=PPFuncSort(args=[PPSortVar(name='B')],
                                                                                        rtpe=PPTensorSort(
                                                                                            param_sort=PPInt(), shape=[
                                                                                                PPDimConst(value=5)]))),
                                                               PPTermNT(name='Z', sort=PPFuncSort(args=[
                                                                   PPTensorSort(param_sort=PPReal(),
                                                                                shape=[PPDimConst(value=5)])],
                                                                   rtpe=PPSortVar(
                                                                       name='B')))])
            )
    printTerm(t)
    # lib.compose((Z: (B --> Tensor[int][5])), (Z: (Tensor[real][5] --> B)))
    t = Rules.expandToVar(libSynth, t, 1, 'nn_fun_2')
    assert (t == PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPVar(name='lib.nn_fun_2'), PPTermNT(name='Z',
                                                                                                    sort=PPFuncSort(
                                                                                                        args=[
                                                                                                            PPTensorSort(
                                                                                                                param_sort=PPReal(),
                                                                                                                shape=[
                                                                                                                    PPDimConst(
                                                                                                                        value=5)])],
                                                                                                        rtpe=PPTensorSort(
                                                                                                            param_sort=PPBool(),
                                                                                                            shape=[
                                                                                                                PPDimConst(
                                                                                                                    value=5)])))])
            );
    printTerm(t)
    # lib.compose(lib.nn_fun_2, (Z: (Tensor[real][5] --> Tensor[bool][5])))

    t = Rules.expandToVar(libSynth, t, 1, 'nn_fun_1')
    assert (
        t == PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPVar(name='lib.nn_fun_2'), PPVar(name='lib.nn_fun_1')])
    )
    printTerm(t)
    # lib.compose(lib.nn_fun_2, lib.nn_fun_1)


def test_count_5s():
    libSynth = getLib()

    tr1_1_28_28 = mkRealTensorSort([1, 1, 28, 28])
    tr1_1 = mkRealTensorSort([1, 1])
    tb1_1 = mkBoolTensorSort([1, 1])

    libSynth.addItems([PPLibItem('recog_5', func(tr1_1_28_28, tb1_1), None), ])

    fn_sort = func(lst(tr1_1_28_28), tr1_1)

    t = PPTermNT("Z", fn_sort)
    assert (t == PPTermNT(name='Z', sort=PPFuncSort(args=[PPListSort(param_sort=PPTensorSort(param_sort=PPReal(),
                                                                                             shape=[PPDimConst(value=1),
                                                                                                    PPDimConst(value=1),
                                                                                                    PPDimConst(
                                                                                                        value=28),
                                                                                                    PPDimConst(
                                                                                                        value=28)]))],
                                                    rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1),
                                                                                                  PPDimConst(
                                                                                                      value=1)]))))
    printTerm(t)
    # (Z: (List[Tensor[real][1,1,28,28]] --> Tensor[real][1,1]))

    t = Rules.expandToFuncApp(libSynth, t, 1, 'compose')
    assert (t == PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPTermNT(name='Z',
                                                                        sort=PPFuncSort(args=[PPSortVar(name='B')],
                                                                                        rtpe=PPTensorSort(
                                                                                            param_sort=PPReal(),
                                                                                            shape=[PPDimConst(value=1),
                                                                                                   PPDimConst(
                                                                                                       value=1)]))),
                                                               PPTermNT(name='Z', sort=PPFuncSort(args=[PPListSort(
                                                                   param_sort=PPTensorSort(param_sort=PPReal(),
                                                                                           shape=[PPDimConst(value=1),
                                                                                                  PPDimConst(value=1),
                                                                                                  PPDimConst(value=28),
                                                                                                  PPDimConst(
                                                                                                      value=28)]))],
                                                                   rtpe=PPSortVar(
                                                                       name='B')))]))
    printTerm(t)
    # lib.compose((Z: (B --> Tensor[real][1,1])), (Z: (List[Tensor[real][1,1,28,28]] --> B)))

    t = Rules.expandToFuncApp(libSynth, t, 1, 'fold')
    assert (t == PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPFuncApp(fn=PPVar(name='lib.fold'), args=[
        PPTermNT(name='Z', sort=PPFuncSort(
            args=[PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]),
                  PPSortVar(name='A')],
            rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))),
        PPTermNT(name='Z', sort=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))]),
                                                               PPTermNT(name='Z', sort=PPFuncSort(args=[PPListSort(
                                                                   param_sort=PPTensorSort(param_sort=PPReal(),
                                                                                           shape=[PPDimConst(value=1),
                                                                                                  PPDimConst(value=1),
                                                                                                  PPDimConst(value=28),
                                                                                                  PPDimConst(
                                                                                                      value=28)]))],
                                                                   rtpe=PPListSort(
                                                                       param_sort=PPSortVar(
                                                                           name='A'))))]))
    printTerm(t)
    # lib.compose(lib.fold((Z: ((Tensor[real][1,1], A) --> Tensor[real][1,1])), (Z: Tensor[real][1,1])), (Z: (List[Tensor[real][1,1,28,28]] --> List[A])))

    t = Rules.expandToUnk(t, 1)
    assert (t == PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPFuncApp(fn=PPVar(name='lib.fold'), args=[
        PPTermUnk(name='Unk', sort=PPFuncSort(
            args=[PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]),
                  PPSortVar(name='A')],
            rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))),
        PPTermNT(name='Z', sort=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))]),
                                                               PPTermNT(name='Z', sort=PPFuncSort(args=[PPListSort(
                                                                   param_sort=PPTensorSort(param_sort=PPReal(),
                                                                                           shape=[PPDimConst(value=1),
                                                                                                  PPDimConst(value=1),
                                                                                                  PPDimConst(value=28),
                                                                                                  PPDimConst(
                                                                                                      value=28)]))],
                                                                   rtpe=PPListSort(
                                                                       param_sort=PPSortVar(
                                                                           name='A'))))]))
    printTerm(t)
    # lib.compose(lib.fold((Unk: ((Tensor[real][1,1], A) --> Tensor[real][1,1])), (Z: Tensor[real][1,1])), (Z: (List[Tensor[real][1,1,28,28]] --> List[A])))

    t = Rules.expandToFuncApp(libSynth, t, 1, 'zeros')
    assert (t == PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPFuncApp(fn=PPVar(name='lib.fold'), args=[
        PPTermUnk(name='Unk', sort=PPFuncSort(
            args=[PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]),
                  PPSortVar(name='A')],
            rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))),
        PPFuncApp(fn=PPVar(name='lib.zeros'), args=[PPTermNT(name='Z', sort=PPDimConst(value=1))])]), PPTermNT(name='Z',
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
                                                                                                                   rtpe=PPListSort(
                                                                                                                       param_sort=PPSortVar(
                                                                                                                           name='A'))))]))
    printTerm(t)
    # lib.compose(lib.fold((Unk: ((Tensor[real][1,1], A) --> Tensor[real][1,1])), lib.zeros((Z: 1))), (Z: (List[Tensor[real][1,1,28,28]] --> List[A])))

    t = Rules.expandDimConst(t, 1)
    assert (t == PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPFuncApp(fn=PPVar(name='lib.fold'), args=[
        PPTermUnk(name='Unk', sort=PPFuncSort(
            args=[PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]),
                  PPSortVar(name='A')],
            rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))),
        PPFuncApp(fn=PPVar(name='lib.zeros'), args=[PPIntConst(value=1)])]), PPTermNT(name='Z', sort=PPFuncSort(args=[
        PPListSort(param_sort=PPTensorSort(param_sort=PPReal(),
                                           shape=[PPDimConst(value=1), PPDimConst(value=1), PPDimConst(value=28),
                                                  PPDimConst(value=28)]))], rtpe=PPListSort(
        param_sort=PPSortVar(name='A'))))]))
    printTerm(t)
    # lib.compose(lib.fold((Unk: ((Tensor[real][1,1], A) --> Tensor[real][1,1])), lib.zeros(1)), (Z: (List[Tensor[real][1,1,28,28]] --> List[A])))

    t = Rules.expandToFuncApp(libSynth, t, 1, 'map')
    assert (t == PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPFuncApp(fn=PPVar(name='lib.fold'), args=[
        PPTermUnk(name='Unk', sort=PPFuncSort(
            args=[PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]),
                  PPSortVar(name='B')],
            rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))),
        PPFuncApp(fn=PPVar(name='lib.zeros'), args=[PPIntConst(value=1)])]), PPFuncApp(fn=PPVar(name='lib.map'), args=[
        PPTermNT(name='Z', sort=PPFuncSort(args=[PPTensorSort(param_sort=PPReal(),
                                                              shape=[PPDimConst(value=1), PPDimConst(value=1),
                                                                     PPDimConst(value=28), PPDimConst(value=28)])],
                                           rtpe=PPSortVar(name='B')))])]))
    printTerm(t)
    # lib.compose(lib.fold((Unk: ((Tensor[real][1,1], B) --> Tensor[real][1,1])), lib.zeros(1)), lib.map((Z: (Tensor[real][1,1,28,28] --> B))))

    t = Rules.expandToVar(libSynth, t, 1, 'recog_5')
    assert (t == PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPFuncApp(fn=PPVar(name='lib.fold'), args=[
        PPTermUnk(name='Unk', sort=PPFuncSort(
            args=[PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]),
                  PPTensorSort(param_sort=PPBool(), shape=[PPDimConst(value=1), PPDimConst(value=1)])],
            rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))),
        PPFuncApp(fn=PPVar(name='lib.zeros'), args=[PPIntConst(value=1)])]), PPFuncApp(fn=PPVar(name='lib.map'),
                                                                                       args=[PPVar(
                                                                                           name='lib.recog_5')])]))
    printTerm(t)
    # lib.compose(lib.fold((Unk: ((Tensor[real][1,1], Tensor[bool][1,1]) --> Tensor[real][1,1])), lib.zeros(1)), lib.map(lib.recog_5))


###################

def test_sum_digits():
    libSynth = getLib()

    input_type = mkListSort(mkRealTensorSort([1, 1, 28, 28]))
    output_type = mkRealTensorSort([1, 1])
    fn_sort = mkFuncSort(input_type, output_type)

    tr1_1_28_28 = mkRealTensorSort([1, 1, 28, 28])
    tb_1_10 = mkBoolTensorSort([1, 10])
    classify_digit = mkFuncSort(tr1_1_28_28, tb_1_10)

    libSynth.addItems([PPLibItem('classify_digit', classify_digit, None), ])

    t = PPTermNT("Z", fn_sort)
    assert (t == PPTermNT(name='Z', sort=PPFuncSort(args=[PPListSort(param_sort=PPTensorSort(param_sort=PPReal(),
                                                                                             shape=[PPDimConst(value=1),
                                                                                                    PPDimConst(value=1),
                                                                                                    PPDimConst(
                                                                                                        value=28),
                                                                                                    PPDimConst(
                                                                                                        value=28)]))],
                                                    rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1),
                                                                                                  PPDimConst(
                                                                                                      value=1)]))))
    printTerm(t)
    # (Z: (List[Tensor[real][1,1,28,28]] --> Tensor[real][1,1]))

    t = Rules.expandToFuncApp(libSynth, t, 1, 'compose')
    assert (t == PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPTermNT(name='Z',
                                                                        sort=PPFuncSort(args=[PPSortVar(name='B')],
                                                                                        rtpe=PPTensorSort(
                                                                                            param_sort=PPReal(),
                                                                                            shape=[PPDimConst(value=1),
                                                                                                   PPDimConst(
                                                                                                       value=1)]))),
                                                               PPTermNT(name='Z', sort=PPFuncSort(args=[PPListSort(
                                                                   param_sort=PPTensorSort(param_sort=PPReal(),
                                                                                           shape=[PPDimConst(value=1),
                                                                                                  PPDimConst(value=1),
                                                                                                  PPDimConst(value=28),
                                                                                                  PPDimConst(
                                                                                                      value=28)]))],
                                                                   rtpe=PPSortVar(
                                                                       name='B')))]))
    printTerm(t)
    # lib.compose((Z: (B --> Tensor[real][1,1])), (Z: (List[Tensor[real][1,1,28,28]] --> B)))

    t = Rules.expandToFuncApp(libSynth, t, 1, 'fold')
    assert (
        t == PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPFuncApp(fn=PPVar(name='lib.fold'), args=[PPTermNT(name='Z',
                                                                                                               sort=PPFuncSort(
                                                                                                                   args=[
                                                                                                                       PPTensorSort(
                                                                                                                           param_sort=PPReal(),
                                                                                                                           shape=[
                                                                                                                               PPDimConst(
                                                                                                                                   value=1),
                                                                                                                               PPDimConst(
                                                                                                                                   value=1)]),
                                                                                                                       PPSortVar(
                                                                                                                           name='A')],
                                                                                                                   rtpe=PPTensorSort(
                                                                                                                       param_sort=PPReal(),
                                                                                                                       shape=[
                                                                                                                           PPDimConst(
                                                                                                                               value=1),
                                                                                                                           PPDimConst(
                                                                                                                               value=1)]))),
                                                                                                      PPTermNT(name='Z',
                                                                                                               sort=PPTensorSort(
                                                                                                                   param_sort=PPReal(),
                                                                                                                   shape=[
                                                                                                                       PPDimConst(
                                                                                                                           value=1),
                                                                                                                       PPDimConst(
                                                                                                                           value=1)]))]),
                                                           PPTermNT(name='Z', sort=PPFuncSort(args=[PPListSort(
                                                               param_sort=PPTensorSort(param_sort=PPReal(),
                                                                                       shape=[PPDimConst(value=1),
                                                                                              PPDimConst(value=1),
                                                                                              PPDimConst(value=28),
                                                                                              PPDimConst(value=28)]))],
                                                               rtpe=PPListSort(
                                                                   param_sort=PPSortVar(
                                                                       name='A'))))]))

    printTerm(t)
    # lib.compose(lib.fold((Z: ((Tensor[real][1,1], A) --> Tensor[real][1,1])), (Z: Tensor[real][1,1])), (Z: (List[Tensor[real][1,1,28,28]] --> List[A])))

    t = Rules.expandToUnk(t, 1)
    assert (t == PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPFuncApp(fn=PPVar(name='lib.fold'), args=[
        PPTermUnk(name='Unk', sort=PPFuncSort(
            args=[PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]),
                  PPSortVar(name='A')],
            rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))),
        PPTermNT(name='Z', sort=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))]),
                                                               PPTermNT(name='Z', sort=PPFuncSort(args=[PPListSort(
                                                                   param_sort=PPTensorSort(param_sort=PPReal(),
                                                                                           shape=[PPDimConst(value=1),
                                                                                                  PPDimConst(value=1),
                                                                                                  PPDimConst(value=28),
                                                                                                  PPDimConst(
                                                                                                      value=28)]))],
                                                                   rtpe=PPListSort(
                                                                       param_sort=PPSortVar(
                                                                           name='A'))))]))
    printTerm(t)
    # lib.compose(lib.fold((Unk: ((Tensor[real][1,1], A) --> Tensor[real][1,1])), (Z: Tensor[real][1,1])), (Z: (List[Tensor[real][1,1,28,28]] --> List[A])))

    t = Rules.expandToFuncApp(libSynth, t, 1, 'zeros')
    assert (t == PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPFuncApp(fn=PPVar(name='lib.fold'), args=[
        PPTermUnk(name='Unk', sort=PPFuncSort(
            args=[PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]),
                  PPSortVar(name='A')],
            rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))),
        PPFuncApp(fn=PPVar(name='lib.zeros'),
                  args=[PPTermNT(name='Z', sort=PPDimConst(value=1))])]), PPTermNT(name='Z',
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
                                                                                       rtpe=PPListSort(
                                                                                           param_sort=PPSortVar(
                                                                                               name='A'))))]))
    printTerm(t)
    # lib.compose(lib.fold((Unk: ((Tensor[real][1,1], A) --> Tensor[real][1,1])), lib.zeros((Z: 1))), (Z: (List[Tensor[real][1,1,28,28]] --> List[A])))

    t = Rules.expandDimConst(t, 1)
    assert (t == PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPFuncApp(fn=PPVar(name='lib.fold'), args=[
        PPTermUnk(name='Unk', sort=PPFuncSort(
            args=[PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]),
                  PPSortVar(name='A')],
            rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))),
        PPFuncApp(fn=PPVar(name='lib.zeros'), args=[PPIntConst(value=1)])]), PPTermNT(name='Z', sort=PPFuncSort(args=[
        PPListSort(param_sort=PPTensorSort(param_sort=PPReal(),
                                           shape=[PPDimConst(value=1), PPDimConst(value=1), PPDimConst(value=28),
                                                  PPDimConst(value=28)]))], rtpe=PPListSort(
        param_sort=PPSortVar(name='A'))))]))
    printTerm(t)
    # lib.compose(lib.fold((Unk: ((Tensor[real][1,1], A) --> Tensor[real][1,1])), lib.zeros(1)), (Z: (List[Tensor[real][1,1,28,28]] --> List[A])))


    t = Rules.expandToFuncApp(libSynth, t, 1, 'map')
    assert (t == PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPFuncApp(fn=PPVar(name='lib.fold'), args=[
        PPTermUnk(name='Unk', sort=PPFuncSort(
            args=[PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]),
                  PPSortVar(name='B')],
            rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))),
        PPFuncApp(fn=PPVar(name='lib.zeros'), args=[PPIntConst(value=1)])]), PPFuncApp(fn=PPVar(name='lib.map'), args=[
        PPTermNT(name='Z', sort=PPFuncSort(args=[PPTensorSort(param_sort=PPReal(),
                                                              shape=[PPDimConst(value=1), PPDimConst(value=1),
                                                                     PPDimConst(value=28), PPDimConst(value=28)])],
                                           rtpe=PPSortVar(name='B')))])]))
    printTerm(t)
    # lib.compose(lib.fold((Unk: ((Tensor[real][1,1], B) --> Tensor[real][1,1])), lib.zeros(1)), lib.map((Z: (Tensor[real][1,1,28,28] --> B))))


    t = Rules.expandToVar(libSynth, t, 1, 'classify_digit')
    assert (t == PPFuncApp(fn=PPVar(name='lib.compose'), args=[PPFuncApp(fn=PPVar(name='lib.fold'), args=[
        PPTermUnk(name='Unk', sort=PPFuncSort(
            args=[PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]),
                  PPTensorSort(param_sort=PPBool(), shape=[PPDimConst(value=1), PPDimConst(value=10)])],
            rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))),
        PPFuncApp(fn=PPVar(name='lib.zeros'), args=[PPIntConst(value=1)])]), PPFuncApp(fn=PPVar(name='lib.map'), args=[
        PPVar(name='lib.classify_digit')])]))
    printTerm(t)
    # lib.compose(lib.fold((Unk: ((Tensor[real][1,1], Tensor[bool][1,10]) --> Tensor[real][1,1])), lib.zeros(1)), lib.map(lib.classify_digit))


def test_graph_shortest_dist():
    # libSynth = getLib()
    # input_type = mkRealTensorSort([1, 1, 28, 28])
    # output_type = mkBoolTensorSort([1, 10])
    # classify_speed = mkFuncSort(input_type, output_type)
    #
    # libSynth.addItems([PPLibItem('classify_speed', classify_speed, None), ])
    #
    # input_type = mkRealTensorSort([1, 1, 28, 28])
    # output_type = mkRealTensorSort([1, 1])
    # shortest_path_sort = mkFuncSort(input_type, output_type)
    #
    # t = PPTermNT("Z", shortest_path_sort)
    # printTerm(t)
    # t = Rules.expandToFuncApp(libSynth, t, 1, 'compose')
    # assert (t)
    # printTerm(t)
    pass
