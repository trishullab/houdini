from typing import Callable, Optional

from HOUDINI import FnLibrary
from HOUDINI.FnLibrary import PPLibItem
from HOUDINI.Synthesizer import ASTUtils, ASTDSL, ReprUtils, ScopeUtils
from HOUDINI.Synthesizer.AST import *
from HOUDINI.Synthesizer.ASTDSL import mkFuncSort, mkFreshTypeVar
from HOUDINI.Synthesizer.Unification import unifyOne, applySubst

global gUseTypes

gUseTypes = True


def getSortVarNotInTerm(sv, term):
    """
    get a sv that is not in term
    """
    i = 0
    while ASTUtils.exists(term, lambda x: x == sv):
        i = i + 1
        newsv = PPSortVar(sv.name + '_' + str(i))
        sv = newsv

    return sv


def mkDimVarNotInTerm(dv, term):
    """
    get a dv that is not in term
    """
    i = 0
    while ASTUtils.exists(term, lambda x: x == dv):
        i = i + 1
        newdv = PPDimVar(dv.name + '_' + str(i))
        dv = newdv

    return dv


def renameSortVars(libItemSort, term):
    i = 0
    svs = ASTUtils.getSortVars(libItemSort)
    for sv in svs:
        newsv = getSortVarNotInTerm(sv, term)
        libItemSort = ASTUtils.replaceAllSubTerms(libItemSort, sv, newsv)

    return libItemSort


def renameDimVars(libItemSort, term):
    i = 0
    dvs = ASTUtils.getDimVars(libItemSort)
    for dv in dvs:
        newdv = mkDimVarNotInTerm(dv, term)
        libItemSort = ASTUtils.replaceAllSubTerms(libItemSort, dv, newdv)

    return libItemSort


def alphaConvertLibItem(libItem, term):
    libItemSort = renameSortVars(libItem.sort, term)
    libItemSort = renameDimVars(libItemSort, term)

    libItem = PPLibItem(name=libItem.name, sort=libItemSort, obj=libItem.obj)

    return libItem


def expandToVar(lib: FnLibrary, term: PPTerm, ntId: int, vname: str) -> Optional[PPTerm]:
    """
    Generate a new term by replacing a "ntId"th NT from a "term" with a variable (in scope) with name "fname"
    """
    nt = ASTUtils.getNthNT(term, ntId)
    # libItem = ScopeUtils.getAVarInScope(lib, term, ntId, vname)
    libItem = lib.getWithLibPrefix(vname)
    assert libItem

    libItem = alphaConvertLibItem(libItem, term)
    subst = unifyOne(libItem.sort, nt.sort)

    if not gUseTypes:
        if subst is None:
            subst = []

    termExpanded = None
    if subst is not None:
        termUnified = applySubst(subst, term)
        termExpanded = ReprUtils.replaceNthNT(termUnified, ntId, PPVar(libItem.name))

    return termExpanded


def expandToFuncApp(lib: FnLibrary, term: PPTerm, ntId: int, fname: str) -> Optional[PPTerm]:
    resTerm = None
    # Not needed now as there are no lambda terms
    # libItem = ScopeUtils.getAVarInScope(lib, term, ntId, fname)
    libItem = lib.getWithLibPrefix(fname)
    assert libItem
    nt = ASTUtils.getNthNT(term, ntId)

    libItem = alphaConvertLibItem(libItem, term)
    # TODO: expandToVar passed nt.sort as second argument
    subst = unifyOne(nt.sort, libItem.sort.rtpe)

    if not gUseTypes:
        if subst is None:
            subst = []

    if subst is not None:
        nts = [PPTermNT('Z', arg_sort) for arg_sort in libItem.sort.args]
        fnApp = PPFuncApp(PPVar(libItem.name), nts)

        termUnified = applySubst(subst, term)
        fnAppUnified = applySubst(subst, fnApp)

        resTerm = ReprUtils.replaceNthNT(termUnified, ntId, fnAppUnified)

    return resTerm


def expandToUnk(term: PPTerm, ntId: int) -> Optional[PPTerm]:
    """
    Generate a new term by replacing a "ntId"th NT from a "term" with a PPTermUnk
    """
    nt = ASTUtils.getNthNT(term, ntId)

    # Avoid generating Unk of type PPDimConst, PPDimVar, PPEumSort, or PPInt()
    if isinstance(nt.sort, PPDimConst) or isinstance(nt.sort, PPDimVar) or isinstance(nt.sort, PPEnumSort) or \
            isinstance(nt.sort, PPInt):
        return None

    unk = ASTDSL.mkUnk(nt.sort)

    # subst = unifyOne(unk.sort, nt.sort)
    #
    # if subst != []:
    #     print("non empty subst")
    #
    # termExpanded = None
    # if subst is not None:
    #     termUnified = applySubst(subst, term)
    #     termExpanded = ReprUtils.replaceNthNT(termUnified, ntId, unk)

    termNew = ReprUtils.replaceNthNT(term, ntId, unk)
    return termNew


def expandDimConst(term: PPTerm, ntId: int) -> Optional[PPTerm]:
    """
    Expand dimension constant to integer constants (Required for fold zeros)
    """
    nt = ASTUtils.getNthNT(term, ntId)
    if type(nt.sort) != PPDimConst:
        return None

    subTerm = PPIntConst(nt.sort.value)
    termExpanded = ReprUtils.replaceNthNT(term, ntId, subTerm)
    return termExpanded


def expandEnum(term: PPTerm, ntId: int, subTerm: PPTerm) -> Optional[PPTerm]:
    nt = ASTUtils.getNthNT(term, ntId)
    termExpanded = ReprUtils.replaceNthNT(term, ntId, subTerm)
    return termExpanded


# def expandToUnkFuncApp(lib: NewLibrary, term: PPTerm, ntId: int) -> Optional[PPTerm]:
#     """
#     Deprecated.
#     Generate a new term by replacing a "ntId"th NT from a "term" with a (PPTermNT: NewT -> T)(Z: T)
#     """
#     # TODO_deprecated: Also generate unkFuncapp of arity 2, 3, ... (Not needed currently)
#     nt = ASTUtils.getNthNT(term, ntId)
#     ftv = mkFreshTypeVar()
#     # TODO_deprecated: generate PPTermNT if you want to increase the search space.
#     fn = PPTermUnk('Unk', mkFuncSort(ftv, nt.sort))
#     newSubTerm = PPFuncApp(fn, [PPTermNT('Z', ftv)])
#     resTerm = ReprUtils.replaceNthNT(term, ntId, newSubTerm)
#     return resTerm


# def expandToLambda(term: PPTerm, ntId: int) -> Optional[PPTerm]:
#     """
#     Deprecated.
#     """
#     resTerm = None
#     nt = ASTUtils.getNthNT(term, ntId)
#
#     if type(nt.sort) == PPFuncSort:
#         params = list(map(ASTDSL.mkFreshVarDecl, nt.sort.args))
#
#         body = PPTermNT('Z', nt.sort.rtpe)
#         lambdaTerm = PPLambda(params, body)
#
#         resTerm = ReprUtils.replaceNthNT(term, ntId, lambdaTerm)
#
#     return resTerm


def applyExpandToVar(lib: FnLibrary, term: PPTerm, ntId: int) -> List[PPTerm]:
    varDecls = ScopeUtils.getAllVarsInScope(lib, term, ntId)

    res = []
    for var in varDecls:
        nxtTerm = expandToVar(lib, term, ntId, var.name)
        if nxtTerm is not None:
            res.append(nxtTerm)

    return res


def applyExpandToUnk(lib: FnLibrary, term: PPTerm, ntId: int) -> List[PPTerm]:
    # Expand to Unk term
    res = []
    nxtTerm = expandToUnk(term, ntId)
    if nxtTerm is not None:
        res.append(nxtTerm)

    return res


def applyExpandToFuncApp(lib: FnLibrary, term: PPTerm, ntId: int) -> List[PPTerm]:
    varDecls = ScopeUtils.getAllFnVarsInScope(lib, term, ntId)

    res = []
    for var in varDecls:
        nxtTerm = expandToFuncApp(lib, term, ntId, var.name)
        if nxtTerm is not None:
            res.append(nxtTerm)

    # # Expand to unk func app.
    # nxtTerm = expandToUnkFuncApp(lib, term, ntId)
    # if nxtTerm is not None:
    #     res.append(nxtTerm)

    return res


# def applyExpandToLambda(lib: NewLibrary, term: PPTerm, ntId: int) -> List[PPTerm]:
#     nxtTerm = expandToLambda(term, ntId)
#     if nxtTerm is not None:
#         return [nxtTerm]
#     else:
#         return []


def applyExpandEnum(lib: FnLibrary, term: PPTerm, ntId: int) -> List[PPTerm]:
    res = []

    nt = ASTUtils.getNthNT(term, ntId)
    if type(nt.sort) != PPEnumSort:
        return res

    for i in range(nt.sort.start, nt.sort.end + 1):
        subTerm = PPIntConst(i)
        nxtTerm = expandEnum(term, ntId, subTerm)
        if nxtTerm is not None:
            res.append(nxtTerm)

    return res


def applyExpandDimConst(lib: FnLibrary, term: PPTerm, ntId: int) -> List[PPTerm]:
    res = []
    nxtTerm = expandDimConst(term, ntId)
    if nxtTerm is not None:
        res.append(nxtTerm)

    return res


rules = [
    applyExpandToUnk,
    applyExpandToVar,
    applyExpandToFuncApp,
    applyExpandEnum,
    applyExpandDimConst,
]


def getRule(ruleId: int) -> Callable[[PPTerm, int], List[PPTerm]]:
    return rules[ruleId]
