from typing import List

from HOUDINI.Synthesizer import MiscUtils
from HOUDINI.Synthesizer.AST import *


def mkFnApp(fnTerm: PPTerm, args: List[PPTerm]):
    return PPFuncApp(fnTerm, args)


def mkFuncDecl(name, sort: PPFuncSort):
    return PPFuncDecl(PPSymbol(name), sort)


def mkListSort(sort: PPSort):
    return PPListSort(sort)


def mkGraphSort(sort: PPSort):
    return PPGraphSort(sort)


freshVarId = MiscUtils.getUniqueFn()
freshTypeVar = MiscUtils.getUniqueFn()


def mkFreshVarDecl(sort: PPSort) -> PPVarDecl:
    return PPVarDecl('x%s' % freshVarId(), sort)


def mkFreshVarDecls(sorts: List[PPSort]) -> List[PPVar]:
    return map(mkFreshVarDecl, sorts)


def mkFreshTypeVar():
    return PPSortVar('TV%s' % freshTypeVar())


def mkUnkFuncVarDecl(sort: PPSort) -> PPVarDecl:
    return PPVarDecl('unk_fun_%s' % freshVarId(), sort)


def mkNT(sort: PPSort):
    return PPTermNT('Z', sort)


def mkUnk(sort: PPSort):
    return PPTermUnk('Unk', sort)


def mkFuncSort(*sortlist):
    return PPFuncSort(list(sortlist[:-1]), sortlist[-1])


def mkTensorSort(sort, rdims):
    dims = []
    for rdim in rdims:
        if type(rdim) == str:
            dims.append(PPDimVar(rdim))
        elif type(rdim) == int:
            dims.append(PPDimConst(rdim))
        else:
            raise Exception("Unhandled dimension")

    return PPTensorSort(sort, dims)


def mkIntTensorSort(rdims):
    return mkTensorSort(PPInt(), rdims)


def mkRealTensorSort(rdims):
    return mkTensorSort(PPReal(), rdims)


def mkBoolTensorSort(rdims):
    return mkTensorSort(PPBool(), rdims)
