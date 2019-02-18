from typing import Optional, List, Dict

from HOUDINI.FnLibrary import PPLibItem
from HOUDINI.Synthesizer import ASTUtils
from HOUDINI.Synthesizer.AST import *


def alphaConvertVarDecl(vdecl: PPVarDecl, term: PPTerm) -> PPVarDecl:
    """
    Rename sort variables in vdecl to avoid clash with sort variables in the term.
    Should be involved on vdecls in global scope.
    """
    return vdecl


def getAVarInScope(lib, term, ntId, vname) -> Optional[PPVarDecl]:
    vdecl = _getAVarInLocalScope(term, ntId, vname)
    if vdecl is None:
        libItem = lib.get(vname)
        vdecl = PPVarDecl('lib.' + libItem.name, libItem.sort)
        vdecl = alphaConvertVarDecl(vdecl, term)

    return vdecl


def _getAVarInLocalScope(term, ntId, vname) -> Optional[PPVarDecl]:
    localScope = getScopeOfNthNT(term, ntId)
    varDecl = localScope.get(vname)
    return varDecl


def getAllVarsInScope(lib, term, ntId) -> List[PPVarDecl]:
    # TODO: Alpha convert sort variables in global varDecls
    newScope = lib.getDict()

    # This is not needed now as there are no lambda terms in the new grammar
    # localscope = getScopeOfNthNT(term, ntId)
    # newScope.update(localscope)
    return [v for (k, v) in newScope.items()]


def getAllFnVarsInScope(lib, term, ntId) -> List[PPVarDecl]:
    allVars = getAllVarsInScope(lib, term, ntId)
    fnVars = [varDecl for varDecl in allVars if type(varDecl.sort) == PPFuncSort]
    return fnVars


def getScopeOfNthNT(term: PPTerm, n: int) -> Dict[str, PPVarDecl]:
    """
    Returns context variables for nth NT. Assumes that nth NT exists in the given term
    """
    varDecls: Dict[str, PPVarDecl] = None
    count = 0
    found = False

    def traverse(iterm: PPTerm, ivars: Dict[str, PPVarDecl]):
        nonlocal count, varDecls, found

        if type(iterm) == PPTermNT:
            count += 1
            if count == n:
                varDecls = ivars
                found = True
        elif type(iterm) == PPLambda:
            iVarsNew = ivars.copy()
            for param in iterm.params:
                iVarsNew[param.name] = param

            traverse(iterm.body, iVarsNew)
        else:
            for c in ASTUtils.deconstruct(iterm):
                if found:
                    break
                traverse(c, ivars)

    traverse(term, {})

    return varDecls
