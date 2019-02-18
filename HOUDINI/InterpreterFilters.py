from typing import Tuple

from HOUDINI.Interpreter.NeuralModules import NetMLP
from HOUDINI.Synthesizer import ASTUtils
from HOUDINI.Synthesizer.AST import *
from HOUDINI.Synthesizer.ASTUtils import isArgumentOfFun, deconstruct


def get_lib_names(term: PPTerm) -> List[str]:
    if isinstance(term, PPVar):
        name = term.name
        if name[:4] == "lib.":
            return [name[4:]]
        else:
            return []
    else:
        nts = []
        for c in deconstruct(term):
            cnts = get_lib_names(c)
            nts.extend(cnts)
        return nts


def is_evaluable(st, ns) -> Tuple[bool, int]:
    # The program should not be open (no non-terminals).
    if ASTUtils.isOpen(st):
        return False, 1

    unks = ASTUtils.getUnks(st)

    # At most 3 unks for now.
    if len(unks) > 3:
        return False, 2

    # return False, 3
    # print(repr_py(st))

    number_of_mlp_nns = 0

    for unk in unks:
        # type variables and dimension variables not allowed.
        if ASTUtils.isAbstract(unk.sort):
            return False, 3

        # Only function types allowed
        if type(unk.sort) != PPFuncSort:
            return False, 4

        # ******* INPUTS ******* :
        # An input to a function can't be a function
        if any([type(arg_sort) == PPFuncSort for arg_sort in unk.sort.args]):
            return False, 11

        fn_input_sort = unk.sort.args[0]
        fn_output_sort = unk.sort.rtpe

        # No more than 2 arguments
        num_input_arguments = unk.sort.args.__len__()
        if num_input_arguments > 1:
            in1_is_2d_tensor = type(unk.sort.args[0]) == PPTensorSort and unk.sort.args[0].shape.__len__() == 2
            in2_is_2d_tensor = type(unk.sort.args[1]) == PPTensorSort and unk.sort.args[1].shape.__len__() == 2
            out_is_2d_tensor = type(unk.sort.rtpe) == PPTensorSort and unk.sort.rtpe.shape.__len__() == 2
            # If a function takes 2 inputs, they'll be concatenated. Thus, we need them to be 2 dimensional tensors
            if num_input_arguments == 2 and in1_is_2d_tensor and in2_is_2d_tensor and out_is_2d_tensor:
                continue
            else:
                return False, 5

        # If the NN's input is a list, it should be: List<2dTensor> -> 2dTensor
        # (as seq-to-seq models aren't supported)
        # We support List<2dTensor> -> 2dTensor
        if type(unk.sort.args[0]) == PPListSort:
            in_is_list_of_2d_tensors = type(unk.sort.args[0].param_sort) == PPTensorSort and len(unk.sort.args[0].param_sort.shape) == 2
            out_is_2d_tensor = type(unk.sort.rtpe) == PPTensorSort and unk.sort.rtpe.shape.__len__() == 2

            if in_is_list_of_2d_tensors and out_is_2d_tensor:
                continue
            else:
                return False, 6

        # If the input to the NN is an image:
        cnn_feature_dim = 64
        input_is_image = type(fn_input_sort) == PPTensorSort and fn_input_sort.shape.__len__() == 4
        if input_is_image:
            # if the input is of size [batch_size, _, 28, 28], the output must be of size [batch_size, 32, 4, 4]
            cond0a1 = fn_input_sort.shape[2].value == 28 and fn_input_sort.shape[3].value == 28
            cond0a2 = type(fn_output_sort) == PPTensorSort and fn_output_sort.shape.__len__() == 4 and \
                      fn_output_sort.shape[1].value == cnn_feature_dim \
                      and fn_output_sort.shape[2].value == 4 and fn_output_sort.shape[3].value == 4
            # if the input is of size [batch_size, 32, 4, 4], the output must be two dimensional
            cond0b1 = fn_input_sort.shape[1].value == cnn_feature_dim \
                      and fn_input_sort.shape[2].value == 4 and fn_input_sort.shape[3].value == 4
            cond0b2 = type(fn_output_sort) == PPTensorSort and fn_output_sort.shape.__len__() == 2

            if not ((cond0a1 and cond0a2) or (cond0b1 and cond0b2)):
                return False, 50

            if cond0b1 and cond0b2:
                number_of_mlp_nns += 1
            continue

        # if the input is a 2d tensor:
        in_is_2d_tensor = type(unk.sort.args[0]) == PPTensorSort and unk.sort.args[0].shape.__len__() == 2
        out_is_2d_tensor = type(unk.sort.rtpe) == PPTensorSort and unk.sort.rtpe.shape.__len__() == 2
        if in_is_2d_tensor:
            if out_is_2d_tensor:
                number_of_mlp_nns += 1
            else:
                return False, 51

        return False, 52

    lib_names = get_lib_names(st)
    # don't allow multiple repeats, as we could just keep on stacking these.
    if lib_names.count("repeat") > 1:
        return False, 15

    # lib_names = self.interpreter
    for lib_name in lib_names:
        # TODO: check if the functions is part of graph convolution, but is used outside of a conv_g operator.

        if type(ns.lib.items[lib_name].obj) == NetMLP:
            number_of_mlp_nns += 1

    # don't allow for multiple MLPs, as we can just keep on stacking them
    # (thus going into architecture search, which is out of scope)
    if number_of_mlp_nns > 1:
        return False, 16

    return True, 0