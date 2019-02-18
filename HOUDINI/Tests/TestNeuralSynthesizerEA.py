from HOUDINI.Eval.EvaluatorUtils import get_io_examples_count_digit_occ, iterate_diff_training_sizes
from HOUDINI.Interpreter.Interpreter import Interpreter
from HOUDINI.FnLibraryFunctions import get_items_from_repo
from HOUDINI.NeuralSynthesizerEA import NeuralSynthesizerEA, NeuralSynthesizerEASettings
from HOUDINI.FnLibrary import FnLibrary, PPLibItem
from HOUDINI.Synthesizer.AST import PPFuncApp, PPVar, PPTermUnk, PPFuncSort, PPTensorSort, PPReal, PPDimConst, \
    PPBool, PPIntConst
from HOUDINI.Synthesizer.ASTDSL import mkRealTensorSort, mkBoolTensorSort, mkFuncSort, mkListSort
from HOUDINI.Synthesizer.SymbolicSynthesizerEA import SymbolicSynthesizerEA


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


def testNeuralSynthesizerEA():
    lib = getCountFiveLib()
    epochs = 2
    interpreter = Interpreter(lib, epochs)
    sort = getCountFiveSort()
    synth = SymbolicSynthesizerEA(lib, sort)

    def createSettings():
        G = 4  # Generations
        M = 100  # Evaluation limit
        K = 5  # Report top-K programs
        return NeuralSynthesizerEASettings(G, M, K)

    settings = createSettings()

    nsynth = NeuralSynthesizerEA(interpreter, synth, lib, sort, settings)

    digit = 5
    tsize = 120
    vsize = 100
    tio, vio = get_io_examples_count_digit_occ(digit, tsize, vsize)
    for i, (ctio, _) in enumerate(iterate_diff_training_sizes(tio, [50, 100])):
        res = nsynth.solve(ctio, vio)
        print(res.top_k_solutions_results)

