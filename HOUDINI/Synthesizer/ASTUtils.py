from functools import wraps
from typing import Callable, Optional, Dict, Tuple

from HOUDINI.Synthesizer import MiscUtils, Unification
from HOUDINI.Synthesizer.AST import *
from HOUDINI.Synthesizer.AST import PPTerm
from HOUDINI.Synthesizer.Unification import substOne

TreeNodeSort = Union[PPTerm, PPSort, PPVarDecl, tuple, list]


def applyTd(term: TreeNodeSort,
            cond: Callable[[TreeNodeSort], bool],
            func: Callable[[TreeNodeSort], TreeNodeSort]) -> TreeNodeSort:
    if cond(term):  # Found the term
        res = func(term)
    else:
        # Term not found. Go down the tree
        childs = deconstruct(term)
        newChilds = [applyTd(c, cond, func) for c in childs]
        newTerm = construct(term, newChilds)
        res = newTerm

    return res


# Commented as it is not tested.
# def applyTdB(term: TreeNodeSort,
#              func: Callable[[TreeNodeSort], Tuple[bool, TreeNodeSort]]) -> TreeNodeSort:
#     (applied, res) = func(term)
#
#     if not applied:
#         childs = deconstruct(term)
#         newChilds = [applyTdB(c, func)[1] for c in childs]
#         newTerm = construct(term, newChilds)
#         res = newTerm
#         return res
#     else:
#         return res


def applyTdOnce(term: TreeNodeSort,
                cond: Callable[[TreeNodeSort], bool],
                func: Callable[[TreeNodeSort], TreeNodeSort]) -> TreeNodeSort:
    applied = False

    def applyTdOnceRec(termRec: TreeNodeSort) -> TreeNodeSort:
        nonlocal applied

        if applied:
            return termRec

        if cond(termRec):  # Found the term
            res = func(termRec)
            applied = True
        else:
            # Term not found. Go down the tree
            childs = deconstruct(termRec)
            newChilds = [applyTdOnceRec(c) for c in childs]
            newTerm = construct(termRec, newChilds)
            res = newTerm

        return res

    return applyTdOnceRec(term)


def applyTdProg(term: PPTerm,
            cond: Callable[[PPTerm], bool],
            func: Callable[[PPTerm], PPTerm]) -> PPTerm:
    if cond(term):  # Found the term
        res = func(term)
    else:
        # Term not found. Go down the tree
        childs = deconstructProg(term)
        newChilds = [applyTdProg(c, cond, func) for c in childs]
        newTerm = constructProg(term, newChilds)
        res = newTerm

    return res


def applyTdProgGeneral(term: PPTerm,
                       func: Callable[[PPTerm], PPTerm]) -> PPTerm:
    res = None
    newTerm = func(term)
    if newTerm is None:  # process subterms
        childs = deconstructProg(term)
        newChilds = [applyTdProgGeneral(c, func) for c in childs]
        newTerm = constructProg(term, newChilds)
        res = newTerm
    elif newTerm is not None:  # replace term and proceed to sibling
        res = newTerm

    return res

def deconstructProg(prog: PPTerm):
    """
    Infers type of 'prog' in bottom-up fashion.
    """
    if isinstance(prog, PPTermUnk):
        return []
    elif isinstance(prog, PPVar):
        return []
    elif isinstance(prog, PPFuncApp):
        return prog.args
    elif isinstance(prog, PPIntConst):
        return []
    elif isinstance(prog, PPRealConst):
        return []
    elif isinstance(prog, PPBoolConst):
        return []
    else:
        raise NotImplementedError()


def constructProg(prog: PPTerm, newChilds):
    if isinstance(prog, PPTermUnk):
        return prog
    elif isinstance(prog, PPVar):
        return prog
    elif isinstance(prog, PPFuncApp):
        oldChilds = prog.args

        modified = False
        if len(newChilds) == len(oldChilds):
            for newChild, oldChild in zip(newChilds, oldChilds):
                if id(newChild) != id(oldChild):
                    modified = True
                    break
        else:
            modified = True

        if not modified:
            return prog
        else:
            return PPFuncApp(prog.fn, newChilds)

    elif isinstance(prog, PPIntConst):
        return prog
    elif isinstance(prog, PPRealConst):
        return prog
    elif isinstance(prog, PPBoolConst):
        return prog
    else:
        raise NotImplementedError()


def exists(term: TreeNodeSort, cond: Callable[[TreeNodeSort], bool]) -> bool:
    if cond(term):  # Term found.
        res = True
    else:  # Term not found. Go down the tree
        childs = deconstruct(term)
        res = any(exists(c, cond) for c in childs)

    return res


def deconstruct(obj) -> List[TreeNodeSort]:
    if isinstance(obj, (*PPTermTypes, list, tuple)):
        return list(obj.__iter__())
    else:
        # print("type skipped: ", type(obj))
        return []


def construct(obj: TreeNodeSort, childs: List[object]) -> TreeNodeSort:
    def case(sort):
        return type(obj) == sort

    old_childs = deconstruct(obj)

    modified = False
    if len(childs) == len(old_childs):
        for i in range(len(childs)):
            if id(childs[i]) != id(old_childs[i]):
                modified = True
                break
    else:
        modified = True

    if not modified:
        return obj

    if case(PPIntConst):
        return PPIntConst(childs[0])
    elif case(PPRealConst):
        return PPRealConst(childs[0])
    elif case(PPBoolConst):
        return PPBoolConst(childs[0])
    elif case(PPListTerm):
        return PPListTerm(childs[0])
    elif case(PPLambda):
        return PPLambda(childs[0], childs[1])
    elif case(PPFuncApp):
        return PPFuncApp(childs[0], childs[1])
    elif case(PPVar):
        return PPVar(childs[0])
    elif case(PPTermNT):
        return PPTermNT(childs[0], childs[1])
    elif case(PPTermUnk):
        return PPTermUnk(childs[0], childs[1])
    # Sorts
    elif case(PPInt):
        return PPInt()
    elif case(PPBool):
        return PPBool()
    elif case(PPReal):
        return PPReal()
    elif case(PPListSort):
        return PPListSort(childs[0])
    elif case(PPGraphSort):
        return PPGraphSort(childs[0])
    elif case(PPTensorSort):
        return PPTensorSort(childs[0], childs[1])
    elif case(PPFuncSort):
        return PPFuncSort(childs[0], childs[1])
    elif case(PPSortVar):
        return PPSortVar(childs[0])
    # Other
    elif case(PPVarDecl):
        return PPVarDecl(childs[0], childs[1])
    # Python
    elif case(tuple):
        return tuple(childs)
    elif case(list):
        return childs
    else:
        raise Exception('Unhandled type in construct: %s: ' % type(obj))


def isNT(t):
    return isinstance(t, PPNTTypes)


def isUnk(t):
    return isinstance(t, PPTermUnk)


def get_first_nt(term: PPTerm) -> PPNT:
    first_nt = None
    if isNT(term):
        first_nt = term
    else:
        for c in deconstruct(term):
            first_nt = get_first_nt(c)
            if first_nt:
                break

    return first_nt


def getNthNT(term: PPTerm, n: int) -> Optional[PPTermNT]:
    count = 0
    res = None

    def getNthNTRec(termRec: PPTerm):
        nonlocal count, res

        if isNT(termRec):
            count += 1
            if count == n:
                res = termRec
        else:
            for c in deconstruct(termRec):
                getNthNTRec(c)
                if count >= n:
                    break

    getNthNTRec(term)

    return res


def doubleVarNames(term):
    def is_var(x):
        return isinstance(x, PPVar)

    def double_var(x):
        return PPVar(x.name + x.name)

    return applyTd(term, is_var, double_var)


def getNTs(term: PPTerm) -> List[PPTermNT]:
    if isinstance(term, PPNTTypes):
        return [term]
    else:
        nts = []
        for c in deconstruct(term):
            cnts = getNTs(c)
            nts.extend(cnts)
        return nts


def getVars(term: PPTerm) -> List[str]:
    """
    Returns all Vars
    """
    if isinstance(term, PPVar):
        return [term.name]
    else:
        vars = []
        for c in deconstruct(term):
            cvars = getVars(c)
            vars.extend(cvars)
        return vars


def getSortVars(term: PPTerm) -> List[str]:
    """
    Returns all SortVars
    """
    if isinstance(term, PPSortVar):
        return [term.name]
    else:
        sortVars = []
        for c in deconstruct(term):
            cvars = getSortVars(c)
            sortVars.extend(cvars)
        return sortVars


def getUnks(term: PPTerm) -> List[PPTermUnk]:
    if isinstance(term, PPTermUnk):
        return [term]
    else:
        nts = []
        for c in deconstruct(term):
            cnts = getUnks(c)
            nts.extend(cnts)
        return nts


def getNumNTs(term: PPTerm) -> int:
    return len(getNTs(term))


def getNumUnks(term: PPTerm) -> int:
    return len(getUnks(term))


def getSize(term: PPTerm) -> int:
    if type(term) in PPTermTypes:
        size = 1
    else:
        size = 0

    for c in deconstruct(term):
        if type(c) in PPTermTypes or type(c) is list or type(c) is tuple:
            size += getSize(c)

    return size


def getSize2(term: PPTerm) -> int:
    if type(term) is PPFuncApp:
        size = 0
    elif type(term) in PPTermTypes:
        size = 1
    else:
        size = 0

    for c in deconstruct(term):
        if type(c) in PPTermTypes or type(c) is list or type(c) is tuple:
            size += getSize2(c)

    return size


def identity(term: PPTerm):
    return construct(term, deconstruct(term))


def isOpen(term: PPTerm):
    return exists(term, lambda x: type(x) == PPTermNT)


def hasUnk(term: PPTerm):
    return exists(term, lambda x: type(x) == PPTermUnk)


def hasRedundantLambda(term: PPTerm):
    def isRedundantLambda(aTerm):
        if type(aTerm) == PPLambda:
            body = aTerm.body
            paramNames = [p.name for p in aTerm.params]

            if len(aTerm.params) == 1:
                paramName = paramNames[0]
                if type(body) == PPVar and body.name == paramName:
                    # dbgPrint("Ignored Lambda: %s" % ReprUtils.repr_py(aTerm))
                    return True

            if type(body) == PPVar and body.name not in paramNames:
                # dbgPrint("Ignored Lambda: %s" % ReprUtils.repr_py(aTerm))
                return True

        return False

    return exists(term, lambda x: isRedundantLambda(x))


def isNthNT(nt_id: int) -> Callable[[PPTerm], bool]:
    nt_cnt = 0

    def cond(term: PPTerm):
        nonlocal nt_cnt
        if type(term) == PPTermNT:
            nt_cnt += 1
            if nt_cnt == nt_id:
                return True
        return False

    return cond


# def getUnkFns(term: PPTerm) -> List[str]:
#     res = []
#     if isinstance(term, PPFuncApp):
#         if term.fname.value.startswith('unk_fun_'):
#             res.append(term.fname)
#     elif isinstance(term, PPVar):
#         if term.name.startswith('unk_fun_'):
#             res.append(term.name)
#
#     for c in deconstruct(term):
#         i_res = getUnkFns(c)
#         res.extend(i_res)
#
#     return list(set(res))


ntNameGen = MiscUtils.getUniqueFn()


def giveUniqueNamesToNTs(st: PPTerm):
    def rename(nt: PPTermNT):
        return PPTermNT("unk_%d" % ntNameGen(), nt.sort)

    return applyTd(st, isNT, rename)


def giveUniqueNamesToUnks(st: PPTerm):
    def rename(nt: PPTermUnk):
        return PPTermUnk("nn_fun_%d" % ntNameGen(), nt.sort)

    return applyTd(st, isUnk, rename)


def getNTNameSortMap(term):
    retMap = {}

    def id1(nt: PPTermNT):
        nonlocal retMap
        retMap[nt.name] = nt.sort
        return nt

    applyTd(term, isNT, id1)

    return retMap


def getUnkNameSortMap(term) -> Dict[str, PPSort]:
    retMap = {}

    def id1(nt: PPTermUnk):
        nonlocal retMap
        retMap[nt.name] = nt.sort
        return nt

    applyTd(term, isUnk, id1)

    return retMap


def isAbstract(sort: PPSort):
    abstract = False

    def query(x):
        nonlocal abstract
        abstract = True
        return x

    applyTd(sort, lambda x: type(x) == PPDimVar or type(x) == PPSortVar or type(x) == PPEnumSort, query)

    return abstract


def getSortVars(sort):
    svs = []

    def query(x):
        nonlocal svs
        if x not in svs:
            svs.append(x)
        return x

    applyTd(sort, lambda x: type(x) == PPSortVar, query)

    return svs


def getDimVars(sort):
    svs = []

    def query(x):
        nonlocal svs
        if x not in svs:
            svs.append(x)
        return x

    applyTd(sort, lambda x: type(x) == PPDimVar, query)

    return svs


def occursIn(sv: PPSortVar, sort: PPSort) -> bool:
    """
    Check if a variable occurs in a term
    """
    return exists(sort, lambda x: x == sv)


def replaceAllSubTerms(term, subTerm, newSubTerm):
    def action(st):
        return newSubTerm

    def cond(st):
        return st == subTerm

    newTerm = applyTd(term, cond, action)
    return newTerm


def isArgumentOfFun(term: PPTerm, fName: str, unkArgName: str):
    """ Checks if the the term has an occurrance
        where "unkArgName" appears as a direct argument of application of fName
    """

    def cond(sterm):
        if type(sterm) == PPFuncApp:
            cond1 = sterm.fn == PPVar(fName)
            cond2 = any(map(lambda x: type(x) == PPTermUnk and x.name == unkArgName, sterm.args))
            return cond1 and cond2
        else:
            return False

    return exists(term, cond)


def alphaConvertSorts(sortsA, sortsC):
    """
    Renames sort and dim variables in 'sortsA'
    to avoid clash with sort and dim variables in sorts
    TODO: Also alphaconvert dimensions.
    """
    def getSortVarsMulti(sortList):
        res = []

        for s in sortList:
            sVars = getSortVars(s)
            res.extend(sVars)

        res = list(set(res))
        return res

    svsA = getSortVarsMulti(sortsA)
    svsC = getSortVarsMulti(sortsC)

    aMap = {}
    svsB = []
    for sva in svsA:
        if sva in svsC:
            newSV = sva
            i = 0
            while newSV in svsA or newSV in svsB or newSV in svsC:
                newSV = PPSortVar(sva.name + str(i))
                i += 1

            aMap[sva] = newSV
            svsB.append(newSV)

    # print(aMap)

    newSortsA = []
    for sa in sortsA:
        nsa = sa
        for key, value in aMap.items():
            nsa = substOne(key, value, nsa)
        newSortsA.append(nsa)

    return newSortsA


# def logInferType(pInferType):
#     @wraps(pInferType)
#     def impl(prog, lib):
#         sort = pInferType(prog, lib)
#         if sort is None:
#             print('BEGIN inferType')
#             print('Prog: ', str(prog))
#             print('Prog Repr: ', repr_py(prog))
#             print('Sort ', repr_py_sort(sort) if sort else 'None')
#             print('END inferType')
#         return sort
#
#     return impl


# @logInferType
def inferType(prog: PPTerm, lib) -> Optional[PPSort]:
    """
    Infers type of 'prog' in bottom-up fashion.
    Only works for concrete leaf node types.
    """
    if isinstance(prog, PPTermUnk):
        return prog.sort
    elif isinstance(prog, PPVar):
        varName = prog.name.replace('lib.', '')
        varSort = lib.get(varName).sort
        return varSort
    elif isinstance(prog, PPFuncApp):
        args = prog.args

        fnName = prog.fn.name.replace('lib.', '')
        li = lib.get(fnName)

        fnSort = li.sort
        argSorts = fnSort.args
        rtpe = fnSort.rtpe
        argSortsConcrete = []
        for arg, argSort in zip(args, argSorts):
            if isinstance(argSort, PPDimVar):
                if not isinstance(arg, PPIntConst):
                    print('prog: ', prog)
                    print('arg: ', arg)
                    raise ValueError('DimVar arg is not of type PPIntConst')

                ct = PPDimConst(arg.value)
            elif isinstance(argSort, PPEnumSort):
                if not isinstance(arg, PPIntConst):
                    print('prog: ', prog)
                    print('arg: ', arg)
                    raise ValueError('Enum arg is not of type PPIntConst')
                ct = argSort
            else:
                ct = inferType(arg, lib)

            argSortsConcrete.append(ct)

        # Rename argument sort vars to avoid conflict
        sortsToRename = list(argSorts)
        sortsToRename.append(rtpe)
        renamedSorts = alphaConvertSorts(sortsToRename, argSortsConcrete)
        argSorts = renamedSorts[:-1]
        rtpe = renamedSorts[-1]

        subst = Unification.unifyLists(argSorts, argSortsConcrete)
        if subst is not None:
            concreteRtpe = Unification.applySubst(subst, rtpe)
        else:
            concreteRtpe = None
        return concreteRtpe
    elif isinstance(prog, PPIntConst):
        return PPInt()
    elif isinstance(prog, PPRealConst):
        return PPReal()
    elif isinstance(prog, PPBoolConst):
        return PPBool()
    elif isinstance(prog, PPListTerm):
        raise NotImplementedError()
    else:
        raise NotImplementedError()


def progTreeSize(prog: PPTerm) -> int:
    """
    Size of the 'prog' where only PPTerm nodes are counted.
    """
    if isinstance(prog, PPFuncApp):
        args = prog.args
        size = 1 + sum([progTreeSize(arg) for arg in args])
        return size
    elif isinstance(prog, PPTermUnk):
        return 1
    elif isinstance(prog, PPVar):
        return 1
    elif isinstance(prog, PPIntConst):
        return 1
    elif isinstance(prog, PPRealConst):
        return 1
    elif isinstance(prog, PPBoolConst):
        return 1
    elif isinstance(prog, PPListTerm):
        raise NotImplementedError()
    else:
        raise NotImplementedError()


def progDepth(prog: PPTerm) -> int:
    if isinstance(prog, PPFuncApp):
        args = prog.args
        depth = 1 + max([progDepth(arg) for arg in args])
        return depth
    elif isinstance(prog, PPTermUnk):
        return 1
    elif isinstance(prog, PPVar):
        return 1
    elif isinstance(prog, PPIntConst):
        return 1
    elif isinstance(prog, PPRealConst):
        return 1
    elif isinstance(prog, PPBoolConst):
        return 1
    elif isinstance(prog, PPListTerm):
        raise NotImplementedError()
    else:
        raise NotImplementedError()



def main():
    # term = PPVar(name='x')
    # print(term)
    # print(doubleVarNames(term))
    # exit()
    # for term in TermsRepo.termsrepo:
    #     print('############')
    #     print(term)
    #     print(doubleVarNames(term))
    pass


if __name__ == '__main__':
    main()
