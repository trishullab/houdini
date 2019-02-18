from HOUDINI.FnLibraryFunctions import get_items_from_repo
from HOUDINI.FnLibrary import FnLibrary, PPLibItem
from HOUDINI.Synthesizer.ASTDSL import *
from HOUDINI.Synthesizer.ReprUtils import repr_py
from HOUDINI.Synthesizer.SymbolicSynthesizer import SymbolicSynthesizer


def mkCountDigitProgram():
    input_type = mkRealTensorSort([1, 1, 28, 28])
    output_type = mkBoolTensorSort([1, 1])
    fn_sort = mkFuncSort(input_type, output_type)


def mkDefaultLib():
    lib = FnLibrary()
    lib.addItems(get_items_from_repo(['compose', 'repeat', 'map_l', 'fold_l', 'conv_l', 'zeros']))
    return lib


def synthesizeCountDigitProgram():
    lib = mkDefaultLib()

    inType = mkRealTensorSort([1, 1, 28, 28])
    outType = mkBoolTensorSort([1, 1])
    recogDigitType = mkFuncSort(inType, outType)

    lib.addItem(PPLibItem('recogFive', recogDigitType, None))

    inType = mkListSort(mkRealTensorSort([1, 1, 28, 28]))
    outType = mkRealTensorSort([1, 1])
    countDigitType = mkFuncSort(inType, outType)

    synth = SymbolicSynthesizer(lib, countDigitType, nnprefix='x')
    for i, (prog, unks) in enumerate(synth.genProgs()):
        pstr = repr_py(prog)
        if 'fold' in pstr:
            if 'recogFive' in pstr:
                if 'zero' in pstr:
                    print(i, pstr)
                    print(i, prog)
                    print(i, unks)

    # lib.compose(lib.fold_l(nn_fun_x_906, lib.zeros(1)), lib.map_l(lib.recogFive))

    # {'nn_fun_x_906': PPFuncSort(
    #     args=[PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]),
    #           PPTensorSort(param_sort=PPBool(), shape=[PPDimConst(value=1), PPDimConst(value=1)])],
    #     rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))}








def main():
    synthesizeCountDigitProgram()


if __name__ == '__main__':
    main()
