# How to derive from a named tuple

import numpy as np
import torch
from HOUDINI.NeuralSynthesizer import NeuralSynthesizer
from matplotlib import pyplot as plt
from torch.autograd import Variable

from HOUDINI.Eval.EvaluatorUtils import get_io_examples_classify_digits
from HOUDINI.Interpreter.Interpreter import Interpreter
from HOUDINI.FnLibraryFunctions import get_items_from_repo
from HOUDINI.FnLibrary import FnLibrary, PPLibItem
from HOUDINI.Synthesizer import ReprUtils, Rules, ASTUtils
from HOUDINI.Synthesizer.AST import *
from HOUDINI.Synthesizer.ASTDSL import mkTensorSort, mkFuncSort, mkListSort, mkRealTensorSort, \
    mkBoolTensorSort, mkIntTensorSort
from HOUDINI.Synthesizer.ReprUtils import repr_py, repr_py_ann
from HOUDINI.Synthesizer.SymbolicSynthesizer import SymbolicSynthesizer


def app_test1():
    NT1 = NamedTuple("NT1", [('a', int), ('b', bool)])

    NT2 = NamedTuple("NT2", [(k, v) for k, v in NT1._field_types.items()] + [('c', str)])

    x = NT1(a=5, b=True)
    y = NT2(a=4, b=False, c="hello")

    print(x)
    print(y)


def app_test2():
    xs = [3, '+', 4, '*', 2, '+', 3, '*', 6]

    def split(ys):
        return [(xs[i], xs[i + 1], xs[i + 2]) for i in range(0, len(xs) - 2)]

    def apply_mul(ys):
        for (a, opr, c) in split(ys):
            if opr == '+':
                return a + c
            if opr == '-':
                pass
            else:
                pass

    xs.map(apply_mul)


def app_test3():
    npx = np.array([0.0, 0.5, 1, 1.5, 2.0]) * np.pi
    x = Variable(torch.Tensor(npx), requires_grad=True)
    y = torch.sin(x)
    x.grad
    y.backward(torch.Tensor([1., 1., 1., 1., 1.]))
    x.grad.data.int().numpy()
    torch.cos(x).data.int().numpy()


def app_test4():
    x = Variable(torch.Tensor(np.linspace(-2, 4, 100)), requires_grad=True)
    y = x ** 2 - 2 * x - 3
    target = -1
    ind = np.where(x.data.numpy() >= target)[0][0]
    gradients = torch.zeros(100, 1)
    gradients[ind] = 1
    y.backward(gradients)

    x_val = float(x[ind].data.numpy())
    y_val = float(y[ind].data.numpy())
    m = float(x.grad.data.numpy()[ind])
    m
    y_tangent = m * (x - x_val) + y_val
    x = x.data.numpy()
    y = y.data.numpy()
    y_tangent = y_tangent.data.numpy()
    plt.plot(x, y, 'g', x[5:30], y_tangent[5:30], 'b', x_val, y_val, 'ro')
    plt.title('$f(x) = x^2 - 2x - 3$')
    plt.show()


def app_test5():
    a = Variable(torch.ones(1, 1) * 2, requires_grad=True)
    b = Variable(torch.ones(1, 1) * 3, requires_grad=True)
    c = Variable(torch.ones(1, 1) * 4, requires_grad=True)

    x = (a + b)
    # Define the function "out" having 2 parameters a,b
    out = (a + b) * c
    # c = torch.mul(a,b)+c
    print('Value out:', out)

    x.backward()
    # Do the backpropagation
    out.backward()

    x.backward()

    print(a.grad)
    print(b.grad)
    print(c.grad)


def printTerm(t):
    print('++++++')
    print(t)
    print('---------')
    print(ReprUtils.repr_py(t))


def func(*lst):
    return mkFuncSort(*lst)


def lst(t):
    return mkListSort(t)


def is_evaluable(st) -> bool:
    def printmsg(msg):
        # print(msg)
        return

    # The program should not be open (no non-terminals).
    if ASTUtils.isOpen(st):
        return False

    unks = ASTUtils.getUnks(st)

    # At most one unk for now.
    if len(unks) > 1:
        printmsg("more than one unk")
        return False

    # print(repr_py(st))


    for unk in unks:
        # type variables and dimension variables not allowed.
        if ASTUtils.isAbstract(unk.sort):
            printmsg("Abstract unk sort")
            return False

        # Only function types allowed
        if type(unk.sort) != PPFuncSort:
            printmsg("unk.sort is not a FuncSort")
            return False

        # NN modules don't take more than 1 input
        if unk.sort.args.__len__() > 1:  # unk.sort returns a PPFuncSort object
            printmsg("unk sort takes more than one arguments")
            return False

        # Sequence output of NNs aren't implemented at the moment
        if type(unk.sort.rtpe) == PPListSort:  # and type(unk.sort.args[0]) == PPListSort (for seq-seq)
            printmsg("unk sort return type is list")
            return False

        # The output type must be a tensor with 2 dimension
        if type(unk.sort.rtpe) != PPTensorSort or type(
                unk.sort.rtpe) == PPTensorSort and unk.sort.rtpe.shape.__len__() != 2:
            printmsg("unk sort return type is not a tensor OR is a tensor but numer of dims is not 2")
            return False

        # Sequence[Images] can't be handled by a RNN atm (todo: double-check, but I think it should be fine)
        if type(unk.sort.args[0]) == PPListSort and unk.sort.args[0].param_sort.shape.__len__() != 2:
            printmsg("unk sort argument type is a list whose number dimension is not 2")
            return False

        if any([type(arg_sort) == PPFuncSort for arg_sort in unk.sort.args]):
            printmsg("One or more argument is of type function")
            return False

    return True


def getLib():
    libSynth = FnLibrary()
    A = PPSortVar('A')
    B = PPSortVar('B')
    C = PPSortVar('C')

    tr5 = mkRealTensorSort([5])
    tb5 = mkBoolTensorSort([5])
    ti5 = mkIntTensorSort([5])
    trab = mkTensorSort(PPReal(), ['a', 'b'])
    tbab = mkTensorSort(PPBool(), ['a', 'b'])

    ppint = PPInt()

    cnts = PPEnumSort(2, 5)

    libSynth.addItems([
        PPLibItem('map', func(func(A, B), func(lst(A), lst(B))), None),
        PPLibItem('fold', func(func(B, A, B), B, func(lst(A), B)), None),
        PPLibItem('conv', func(func(A, lst(A), A), func(lst(A), lst(A))), None),
        PPLibItem('compose', func(func(A, B), func(B, C), func(A, C)), None),
        # PPLibItem('repeat', func(cnts, func(A, A), func(A, A)), None),
        PPLibItem('zeros', func(PPDimVar('a'), mkRealTensorSort([1, 'a'])), None),
        # PPLibItem('nn_fun_0', func(tr5, tr5), None),
        # PPLibItem('nn_fun_1', func(tr5, tb5), None),
        # PPLibItem('nn_fun_2', func(tb5, ti5), None),
        PPLibItem('add', mkFuncSort(trab, trab, trab), None),
        PPLibItem('add1', mkFuncSort(trab, tbab, trab), None),
    ])
    return libSynth


tr5 = mkRealTensorSort([5])
tb5 = mkBoolTensorSort([5])
ti5 = mkIntTensorSort([5])


def app_test():
    libSynth = getLib()

    # Repeat
    t = PPTermNT("Z", func(tr5, tr5))
    assert (t);
    printTerm(t)
    t = Rules.expandToFuncApp(libSynth, t, 1, 'repeat')
    assert (t);
    printTerm(t)
    t = Rules.expandEnum(libSynth, t, 1, PPIntConst(5))
    assert (t);
    printTerm(t)

    return

    # compose
    t = PPTermNT("Z", func(tr5, ti5))
    assert (t);
    printTerm(t)
    t = Rules.expandToFuncApp(libSynth, t, 1, 'compose')
    assert (t);
    printTerm(t)
    t = Rules.expandToVar(libSynth, t, 1, 'nn_fun_1')
    assert (t);
    printTerm(t)
    t = Rules.expandToVar(libSynth, t, 1, 'nn_fun_2')
    assert (t);
    printTerm(t)
    return

    # Unk
    t = PPTermNT("Z", func(lst(tr5), tr5))
    printTerm(t)
    t = Rules.expandToUnk(libSynth, t, 1)
    printTerm(t)

    # fold
    t = PPTermNT("Z", func(lst(tr5), tr5))
    printTerm(t)
    t = Rules.expandToFuncApp(libSynth, t, 1, 'fold')
    printTerm(t)
    t = Rules.expandToFuncApp(libSynth, t, 2, 'zeros')
    printTerm(t)
    t = Rules.expandDimConst(libSynth, t, 2)
    printTerm(t)
    t = Rules.expandToUnk(libSynth, t, 1)
    printTerm(t)

    # map
    t = PPTermNT("Z", func(lst(tr5), lst(tr5)))
    printTerm(t)
    t = Rules.expandToFuncApp(libSynth, t, 1, 'map')
    printTerm(t)
    t = Rules.expandToVar(libSynth, t, 1, 'nn_fun_0')
    printTerm(t)

    # conv
    t = PPTermNT("Z", func(lst(tr5), lst(tr5)))
    printTerm(t)
    t = Rules.expandToFuncApp(libSynth, t, 1, 'conv')
    printTerm(t)
    t = Rules.expandToUnk(libSynth, t, 1)
    assert (t)
    printTerm(t)


def syn_test():
    libSynth = getLib()

    # Recognize Digit
    input_type = mkRealTensorSort([1, 1, 28, 28])
    output_type = mkBoolTensorSort([1, 1])
    fn_sort = mkFuncSort(input_type, output_type)

    libSynth.addItems([PPLibItem('recog_5', fn_sort, None)])

    # Counting
    input_type = mkListSort(mkRealTensorSort([1, 1, 28, 28]))
    output_type = mkRealTensorSort([1, 1])
    fn_sort = mkFuncSort(input_type, output_type)

    synth = SymbolicSynthesizer(libSynth, fn_sort)
    i = 0
    j = 0
    I = 100
    J = 100
    for prog, unkSortMap in synth.genProgs():
        i = i + 1
        if is_evaluable(prog):
            print(repr_py(prog))
            j += 1

        if i > I or j > J:
            break

    print("done")


def main():
    tr1_1_28_28 = mkRealTensorSort([1, 1, 28, 28])
    tr1_1 = mkRealTensorSort([1, 1])
    tb1_1 = mkBoolTensorSort([1, 1])

    libSynth = FnLibrary()
    libSynth.addItems(get_items_from_repo(['compose', 'repeat', 'map_l', 'fold_l', 'conv_l', 'zeros']))

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

        if not NeuralSynthesizer.is_evaluable(prog):
            print(i, "Rejected")
            print(repr_py_ann(prog))
        else:
            print(i, "Accepted")
            print(print(repr_py_ann(prog)))


def main2():
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

    interpreter = Interpreter(lib, 1)
    res = interpreter.evaluate(program=prog,
                               output_type_s=fn_sort.rtpe,
                               unkSortMap=unkSortMap,
                               io_examples_tr=tio,
                               io_examples_val=vio)


if __name__ == '__main__':
    main2()
