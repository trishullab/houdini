from typing import List, Callable

from HOUDINI.Synthesizer.AST import *
from HOUDINI.Synthesizer import ASTUtils


def csv(xs: List[str]) -> str:
    res = ', '.join(xs)
    return res


def repr_py(term: PPTerm) -> str:
    assert term is not None
    term_tpe = type(term)

    def case(tpe):
        return tpe == term_tpe

    if case(PPIntConst):
        return str(term.value)
    elif case(PPRealConst):
        return term.value
    elif case(PPBoolConst):
        return term.value
    elif case(PPVar):
        return term.name
    elif case(PPVarDecl):
        return term.name
    elif case(PPLambda):
        return 'lambda %s: %s' % (csv(map(repr_py, term.params)), repr_py(term.body))
    elif case(PPFuncApp):
        # return '%s(%s)' % (term.fname.value, csv(map(repr_py, term.args)))
        fnRepr = repr_py(term.fn)
        if type(term.fn) != PPVar and type(term.fn) != PPTermUnk:
            fnRepr = '(%s)' % fnRepr
        return '%s(%s)' % (fnRepr, csv(map(repr_py, term.args)))
    elif case(PPListTerm):
        return '[%s]' % csv(map(repr_py, term.items))
    elif case(PPTermNT):
        return term.name
    elif case(PPTermUnk):
        return term.name
    else:
        raise Exception('Unhandled type in printPython: %s' % type(term))


def repr_py_shape(shape: PPDim) -> str:
    res = ''
    if type(shape) == PPDimConst:
        res = str(shape.value)
    elif type(shape) == PPDimVar:
        res = shape.name
    return res


def repr_py_sort(sort: PPSort) -> str:
    def case(tpe):
        return tpe == type(sort)

    if case(PPInt):
        res = 'int'
    elif case(PPReal):
        res = 'real'
    elif case(PPBool):
        res = 'bool'
    elif case(PPListSort):
        res = 'List[%s]' % (repr_py_sort(sort.param_sort),)
    elif case(PPGraphSort):
        res = 'GraphSequences[%s]' % (repr_py_sort(sort.param_sort),)
    elif case(PPTensorSort):
        res = 'Tensor[%s][%s]' % (repr_py_sort(sort.param_sort), ','.join([repr_py_shape(d) for d in sort.shape]))
    elif case(PPFuncSort):
        argsRepr = csv(map(repr_py_sort, sort.args))
        argsRepr = '(%s)' % argsRepr if len(sort.args) > 1 else argsRepr
        res = '(%s --> %s)' % (argsRepr, repr_py_sort(sort.rtpe))
    elif case(PPSortVar):
        res = sort.name
    elif case(PPImageSort):
        res = 'Image'
    elif case(PPEnumSort):
        res = 'EnumSort'
    elif case(PPDimConst) or case(PPDimVar):
        res = repr_py_shape(sort)
    else:
        raise Exception('Unhandled type: %s' % type(sort))

    return res


def repr_py_ann(term: PPTerm) -> str:
    term_tpe = type(term)
    res = ''

    def case(tpe):
        return tpe == term_tpe

    if case(PPIntConst):
        res = str(term.value)
    elif case(PPRealConst):
        res = str(term.value)
    elif case(PPBoolConst):
        res = str(term.value)
    elif case(PPVar):
        res = term.name
    elif case(PPVarDecl):
        res = '%s: %s' % (term.name, term.sort)
    elif case(PPLambda):
        res = 'lambda (%s): %s' % (csv(map(repr_py_ann, term.params)), repr_py_ann(term.body))
    elif case(PPFuncApp):
        fnRepr = repr_py_ann(term.fn)
        if type(term.fn) != PPVar and type(term.fn) != PPTermUnk:
            fnRepr = '(%s)' % fnRepr
        res = '%s(%s)' % (fnRepr, csv(map(repr_py_ann, term.args)))
    elif case(PPListTerm):
        res = '[%s]' % csv(map(repr_py_ann, term.items))
    elif case(PPTermNT):
        res = '(%s: %s)' % (term.name, repr_py_sort(term.sort))
    elif case(PPTermUnk):
        res = '(%s: %s)' % (term.name, repr_py_sort(term.sort))
    else:
        raise Exception('Unhandled type: %s' % term_tpe)

    return res


def expandNthNT(term: PPTerm, ntId: int, expand: Callable[[PPTermNT], PPTerm]) -> PPTerm:
    newTerm = ASTUtils.applyTdOnce(term, ASTUtils.isNthNT(ntId), expand)
    return newTerm


def replaceNthNT(term: PPTerm, ntId: int, newSubTerm: PPTerm) -> PPTerm:
    newTerm = ASTUtils.applyTdOnce(term, ASTUtils.isNthNT(ntId), lambda nt: newSubTerm)
    return newTerm


def simplerep(sort: PPSort):
    # PPSort = Union[PPInt, PPReal, PPBool, PPSortVar, PPListSort, PPGraphSort, PPTensorSort, PPFuncSort, PPImageSort]
    def case(tpe):
        return type(sort) == tpe

    if case(PPInt):
        return 'int'
    elif case(PPReal):
        return 'real'
    elif case(PPBool):
        return 'bool'
    elif case(PPSortVar):
        return sort.name
    elif case(PPDimVar):
        return sort.name
    elif case(PPDimConst):
        return str(sort.value)
    elif case(PPListSort):
        return 'List[%s]' % simplerep(sort.param_sort)
    elif case(PPGraphSort):
        return 'GraphSequences[%s]' % simplerep(sort.param_sort)
    elif case(PPTensorSort):
        param_sort = simplerep(sort.param_sort)
        shape = ', '.join([simplerep(d) for d in sort.shape])
        return 'Tensor[%s][%s]' % (param_sort, shape)
    elif case(PPFuncSort):
        args = ' * '.join([simplerep(a) for a in sort.args])
        ret = simplerep(sort.rtpe)
        return '( %s -> %s)' % (args, ret)
    elif case(PPImageSort):
        return 'Image'
    elif case(PPEnumSort):
        raise NotImplementedError()
