from typing import Optional, Tuple

from HOUDINI.Synthesizer.AST import *
from HOUDINI.Synthesizer import ASTUtils

"""
Invariant for substitution: no id on a lhs occurs in any term earlier in the list
"""

PPSortOrDimVar = Union[PPSortVar, PPDimVar]
PPSortOrDim = Union[PPSort, PPDim]

Substitution = List[Tuple[PPSortOrDimVar, PPSortOrDim]]


def substOne(a: PPSortOrDimVar, b: PPSortOrDim, sortTerm: PPSort) -> PPSort:
    """
    substitute b for all the occurrences of variable a in sortTerm
    """
    term1 = ASTUtils.applyTd(sortTerm, lambda x: type(x) == PPSortVar, lambda x: b if x == a else x)
    term2 = ASTUtils.applyTd(term1, lambda x: type(x) == PPDimVar, lambda x: b if x == a else x)
    return term2


def applySubst(subst: Substitution, sortTerm: PPSort) -> PPSort:
    """
    Apply a substitution right to left
    """
    curTerm = sortTerm
    for (a, b) in reversed(subst):
        curTerm = substOne(a, b, curTerm)
    return curTerm


def occursIn(sv: PPSortVar, sort: PPSort) -> bool:
    """
    Check if a variable occurs in a term
    """
    return ASTUtils.exists(sort, lambda x: x == sv)


def unifyOne(s: PPSortOrDim, t: PPSortOrDim) -> Optional[Substitution]:
    """ Unify a pair of terms
    """

    def case(ss, tt):
        return isinstance(s, ss) and isinstance(t, tt)

    res = None
    if case(PPSortVar, PPSortVar):
        if s == t:
            res = []
        else:
            res = [(s, t)]
    elif case(PPDimVar, PPDimVar):
        if s == t:
            res = []
        else:
            res = [(s, t)]
    elif case(PPEnumSort, PPEnumSort):
        if s == t:
            res = []
        else:
            # No unification for different enums
            None
    elif case(PPSortVar, PPSortTypes):
        if not occursIn(s, t):
            res = [(s, t)]
        else:
            res = None
    elif case(PPDimVar, PPDimConst):
        res = [(s, t)]
    elif case(PPSortTypes, PPSortVar):
        if not occursIn(t, s):
            res = [(t, s)]
        else:
            res = None
    elif case(PPDimConst, PPDimVar):
        res = [(t, s)]
    elif case(PPInt, PPInt) or case(PPReal, PPReal) or case(PPBool, PPBool) or case(PPImageSort, PPImageSort):
        res = []
    elif case(PPDimConst, PPDimConst) and s == t:
        res = []
    elif case(PPListSort, PPListSort):
        res = unifyOne(s.param_sort, t.param_sort)
    elif case(PPGraphSort, PPGraphSort):
        res = unifyOne(s.param_sort, t.param_sort)
    elif case(PPTensorSort, PPTensorSort):
        res = unifyLists([s.param_sort] + s.shape, [t.param_sort] + t.shape)
    elif case(PPFuncSort, PPFuncSort):
        res = unifyLists(s.args + [s.rtpe], t.args + [t.rtpe])
    else:
        res = None

    return res


def unifyLists(xs: List[PPSortOrDim], ys: List[PPSortOrDim]) -> Optional[Substitution]:
    if len(xs) == len(ys):
        pairs = list(zip(xs, ys))
        res = unify(pairs)
    else:
        res = None
    return res


def unify(pairs: List[Tuple[PPSortOrDim, PPSortOrDim]]) -> Optional[Substitution]:
    """
    Unify a list of pairs
    """
    res = None
    if not pairs:
        res = []
    else:
        (x, y) = pairs[0]
        t = pairs[1:]
        t2 = unify(t)
        if t2 is not None:
            t1 = unifyOne(applySubst(t2, x), applySubst(t2, y))
            if t1 is not None:
                res = t1 + t2
            else:
                res = None
        else:
            res = None
    return res
