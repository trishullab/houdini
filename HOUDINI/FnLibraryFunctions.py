import torch
import torch.nn.functional as F
from functools import reduce

from HOUDINI.FnLibrary import FnLibrary, PPLibItem
from HOUDINI.Interpreter.NeuralModules import NetCNN, SaveableNNModule
from HOUDINI.Synthesizer.AST import PPImageSort, PPInt
from HOUDINI.Synthesizer.ASTDSL import mkTensorSort, mkFuncSort, mkListSort, mkRealTensorSort, mkGraphSort
from HOUDINI.Synthesizer.AST import *
import numpy as np
import pickle
import torch.nn as nn
import os


class NotHandledException(Exception):
    pass


def split(x):
    if type(x) == tuple:
        x = x[1]

    x_list = torch.split(x, split_size=1, dim=1)
    x_list = [torch.squeeze(ii, dim=1) for ii in x_list]

    return x_list


def pp_map2d(fn, iterable):
    if type(iterable) == tuple:
        iterable = iterable[1]

    if isinstance(iterable, torch.autograd.variable.Variable):
        iterable = split(iterable)
        iterable = [split(i) for i in iterable]

    result = [[fn(j) for j in i] for i in iterable]
    return result


def pp_map_g(fn):
    def ret(iterable):
        if type(iterable) == tuple:
            iterable = iterable[1]

        if isinstance(iterable, torch.autograd.variable.Variable):
            iterable = split(iterable)
            iterable = [split(i) for i in iterable]

        result = [[fn(j) for j in i] for i in iterable]
        return result
    return ret


def pp_flatten_2d_list(iterable):
    assert (type(iterable) == list)
    assert (iterable.__len__() > 0)
    assert (type(iterable[0]) == list)
    return [item for innerlist in iterable for item in innerlist]


def pp_cat(a, b):
    if type(a) == tuple:
        a = a[1]
    if type(b) == tuple:
        b = b[1]

    return torch.cat((a, b), dim=1)


def pp_map(fn, iterable):
    if isinstance(iterable, torch.autograd.variable.Variable):
        iterable = split(iterable)
    if iterable.__len__() > 0 and type(iterable[0]) == tuple:
        iterable = [i[1] for i in iterable]
    # using list() to force the map to be evaluated, otherwise it's lazily evaluated
    return list(map(fn, iterable))


def pp_map_list(fn):
    def ret(iterable):
        if isinstance(iterable, torch.autograd.variable.Variable):
            iterable = split(iterable)
        if iterable.__len__() > 0 and type(iterable[0]) == tuple:
            iterable = [i[1] for i in iterable]
        # using list() to force the map to be evaluated, otherwise it's lazily evaluated
        return list(map(fn, iterable))

    return ret


def pp_conv_list(fn):
    def ret(iterable):
        if isinstance(iterable, torch.autograd.variable.Variable):
            iterable = split(iterable)
        if iterable.__len__() > 0 and type(iterable[0]) == tuple:
            iterable = [i[1] for i in iterable]

        if iterable.__len__() == 0:
            return []

        if type(iterable[0]) != torch.autograd.variable.Variable:
            raise NotHandledException

        # zero-pad start and end
        zero = torch.zeros_like(iterable[0])
        iterable.insert(0, zero)
        iterable.append(zero)
        result = []
        for idx in range(1, iterable.__len__() - 1):
            c_arr = [iterable[idx - 1], iterable[idx], iterable[idx + 1]]
            result.append(fn(c_arr))
        return result

    return ret


def pp_conv_graph(fn):
    return lambda x: fn(x)


def pp_reduce_graph(fn):
    raise NotImplementedError


def pp_reduce_list(fn, init=None):
    def ret(iterable):
        if isinstance(iterable, torch.autograd.variable.Variable):
            iterable = split(iterable)
        if iterable.__len__() > 0 and type(iterable[0]) == tuple:
            iterable = [i[1] for i in iterable]
        return reduce(fn, iterable) if init is None else reduce(fn, iterable, init)

    return ret


def pp_repeat(num_times_to_repeat, fn):
    def ret(x):
        for i in range(num_times_to_repeat):
            x = fn(x)
        return x

    return ret


def pp_compose(g, f):
    return lambda x: g(f(x))


def pp_reduce(fn, iterable, initializer=None):
    if isinstance(iterable, torch.autograd.variable.Variable):
        iterable = split(iterable)
    if iterable.__len__() > 0 and type(iterable[0]) == tuple:
        iterable = [i[1] for i in iterable]
    return reduce(fn, iterable) if initializer is None else reduce(fn, iterable, initializer)


def pp_get_zeros(dim):
    """
    Returns zeros of shape [var_x.shape[0], dim]
    Atm, it's used for initializing hidden state
    :return:
    """
    """
    if type(var_x) == tuple:
        var_x = var_x[1]
    """

    zeros = torch.zeros(1, dim)
    if torch.cuda.is_available():
        zeros = zeros.cuda()
    return torch.autograd.variable.Variable(zeros)


def get_multiply_by_range09():
    # zeros = torch.zeros(var_x.data.shape[0], dim)
    range_np = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.], dtype=np.float32).reshape((10, 1))
    range_torch = torch.from_numpy(range_np)
    if torch.cuda.is_available():
        range_torch = range_torch.cuda()
    range_var = torch.autograd.variable.Variable(range_torch)

    def pp_multiply_by_range09(inputs):
        return torch.matmul(inputs, range_var)

    return pp_multiply_by_range09


def argmax(x):
    # x = list(x)
    if type(x) == tuple:
        x = x[1]
    values, indices = torch.max(x, 1, keepdim=True)
    # values, indices = torch.max(input=x, dim=1, keepdim=True)
    # return indices
    indices = indices.float()
    return indices


def pp_add(x, y):
    if type(x) == tuple:
        x = x[1]
    if type(y) == tuple:
        y = y[1]
    return x + y


def mk_recognise_5s():
    res = NetCNN("recognise_5s", input_dim=28, input_ch=1, output_dim=1, output_activation=F.sigmoid)
    res.load('Interpreter/Models/is5_classifier.pth.tar')
    return res


t = PPSortVar('T')
t1 = PPSortVar('T1')
t2 = PPSortVar('T2')


def addImageFunctionsToLibrary(libSynth: FnLibrary, load_recognise_5s=True):
    real_tensor_2d = mkTensorSort(PPReal(), ['a', 'b'])
    bool_tensor_2d = mkTensorSort(PPBool(), ['a', 'b'])
    libSynth.addItems([
        PPLibItem('add', mkFuncSort(real_tensor_2d, real_tensor_2d, real_tensor_2d), pp_add),
        PPLibItem('add1', mkFuncSort(real_tensor_2d, bool_tensor_2d, real_tensor_2d), pp_add),

        PPLibItem('map', mkFuncSort(mkFuncSort(t1, t2), mkListSort(t1), mkListSort(t2)), pp_map),
        PPLibItem('map2d', mkFuncSort(mkFuncSort(t1, t2), mkListSort(mkListSort(t1)), mkListSort(mkListSort(t2))),
                  pp_map2d),
        # question ^ should we transform map's definition into using vectors? is this not enough?
        # we don't know the type of the tensor output, w/o knowing the function.

        # PPLibItem('cat', mkFuncSort(mkTensorSort(PPReal(), ['a', 'b']), mkTensorSort(PPReal(), ['a', 'c']),
        #                            mkTensorSort(PPReal(), ['a', 'd'])), pp_cat),  # TODO: d = b + c
        # Question: can we write 'b+c'? I'm not sure if it's useful
        # Also, the input types don't have to be PPReal, but for not it should suffice to just leave it like this?
        # ^ It can accept a tuple of tensors of different shapes, but maybe we can restrict it to tuple of 2 for now.

        # PPLibItem('zeros', mkFuncSort(PPInt(), mkTensorSort(PPReal(), ['a', 'b']), mkTensorSort(PPReal(), ['a', 'c'])), pp_get_zeros),

        # PPLibItem('zeros', mkFuncSort(PPInt(), PPInt(), mkTensorSort(PPReal(), ['a', 'c'])), pp_get_zeros),
        # 4, [2, 5] -> [2, 4]
        # 7, [2, 5] -> [2, 7]
        # Question: How do we say that the ints are the same number, PPInt() == 'c'
        # Also, The input tensor type doesn't have to be PPReal, can be int or bool as well

        # Also, the input tensor can be of any type, doesn't need to be float
        PPLibItem('zeros', mkFuncSort(PPDimVar('a'), mkRealTensorSort([1, 'a'])), pp_get_zeros),
        PPLibItem('reduce_general', mkFuncSort(mkFuncSort(t, t1, t), mkListSort(t1), t, t), pp_reduce),
        PPLibItem('reduce', mkFuncSort(mkFuncSort(t, t, t), mkListSort(t), t), pp_reduce),
        # pp_get_zeros
        # PPLibItem('reduce_with_init_zeros', mkFuncSort(mkFuncSort(t, t1, t), mkListSort(t1), t), pp_reduce_w_zeros_init),
        # Question : the initializer is only optional. How do we encode this information?

        # The following are just test functions for evaluation, not properly typed.
        # ,PPLibItem('mult_range09', mkFuncSort(mkFuncSort(t, t1, t), mkListSort(t1), t, t), get_multiply_by_range09())
        # ,PPLibItem('argmax', mkFuncSort(mkFuncSort(t, t1, t), mkListSort(t1), t, t), argmax)

        # PPLibItem('split', mkFuncSort(PPImageSort(), mkListSort(PPImageSort())), split),
        # PPLibItem('join', mkFuncSort(mkListSort(PPImageSort()), PPImageSort()), None),

    ])
    if load_recognise_5s:
        libSynth.addItems([PPLibItem('recognise_5s', mkFuncSort(mkTensorSort(PPReal(), ['a', 1, 28, 28]),
                                                                mkTensorSort(PPBool(), ['a', 1])), mk_recognise_5s())])

        # set the neural libraries to evaluation mode
        # TODO: need to make sure we're properly switching between eval and train everywhere
        libSynth.recognise_5s.eval()


lib_items_repo = {}


def add_lib_item(li):
    lib_items_repo[li.name] = li


def add_libitems_to_repo():
    def func(*lst):
        return mkFuncSort(*lst)

    def lst(t):
        return mkListSort(t)

    A = PPSortVar('A')
    B = PPSortVar('B')
    C = PPSortVar('C')

    real_tensor_2d = mkTensorSort(PPReal(), ['a', 'b'])
    bool_tensor_2d = mkTensorSort(PPBool(), ['a', 'b'])

    add_lib_item(PPLibItem('compose', func(func(B, C), func(A, B), func(A, C)), pp_compose))
    add_lib_item(PPLibItem('repeat', func(PPEnumSort(9, 10), func(A, A), func(A, A)), pp_repeat))
    add_lib_item(PPLibItem('map_l', func(func(A, B), func(lst(A), lst(B))), pp_map_list))
    add_lib_item(PPLibItem('fold_l', func(func(B, A, B), B, func(lst(A), B)), pp_reduce_list))
    add_lib_item(PPLibItem('conv_l', func(func(lst(A), B), func(lst(A), lst(B))), pp_conv_list))
    add_lib_item(PPLibItem('zeros', func(PPDimVar('a'), mkRealTensorSort([1, 'a'])), pp_get_zeros))

    def graph(t):
        return mkGraphSort(t)

    add_lib_item(PPLibItem('conv_g', func(func(lst(A), B), func(graph(A), graph(B))), pp_conv_graph))
    add_lib_item(PPLibItem('map_g', func(func(A, B), func(graph(A), graph(B))), pp_map_g))
    add_lib_item(PPLibItem('fold_g', func(func(B, A, B), B, func(graph(A), B)), pp_reduce_graph))

    add_lib_item(PPLibItem('flatten_2d_list', func(func(B, C), func(A, B), func(A, C)), pp_flatten_2d_list))


add_libitems_to_repo()


def get_items_from_repo(item_names: List[str]):
    lib_items = [lib_items_repo[item_name] for item_name in item_names]
    return lib_items


def get_item_from_repo(item_name: str):
    lib_item = lib_items_repo[item_name]
    return lib_item


def get_all_items_from_repo():
    return lib_items_repo.values


def loadLibrary(libDirPath):
    libFilePath = libDirPath + '/' + 'lib.pickle'

    if not os.path.isfile(libFilePath):
        return None

    with open(libFilePath, 'rb') as fh:
        libDict = pickle.load(fh)

    lib = FnLibrary()
    for name, (li, isNN) in libDict.items():
        if isNN:
            modelFilePath = libDirPath + '/' + name + '.pth'
            obj = SaveableNNModule.create_and_load(libDirPath, name)
            lib.addItem(PPLibItem(name, li, obj))
        else:
            lib.addItem(get_item_from_repo(name))

    return lib


def loadLibrary1(libDirPath, taskid):
    libFilePath = libDirPath + '/' + 'lib{}.pickle'.format(taskid-1)

    # return None
    if not os.path.isfile(libFilePath):
        return None

    with open(libFilePath, 'rb') as fh:
        libDict = pickle.load(fh)

    lib = FnLibrary()
    for name, (li, isNN) in libDict.items():
        if isNN:
            modelFilePath = libDirPath + '/' + name + '.pth'
            obj = SaveableNNModule.create_and_load(libDirPath, name)
            lib.addItem(PPLibItem(name, li, obj))
        else:
            lib.addItem(get_item_from_repo(name))

    return lib
