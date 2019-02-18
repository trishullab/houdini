from typing import Iterable, Tuple, Dict

import torch.nn.functional as F

from HOUDINI.Synthesizer.ASTDSL import mkRealTensorSort, mkBoolTensorSort
from HOUDINI.Synthesizer.IntermediateTypeHandler import instantiateSortVar
from HOUDINI.FnLibrary import FnLibrary
from HOUDINI.Synthesizer import ASTUtils, Rules, MiscUtils
from HOUDINI.Synthesizer.AST import *
from HOUDINI.Synthesizer.ReprUtils import repr_py
from HOUDINI.Synthesizer.SolverPQ import SolverPQ

Action = NamedTuple("Action", [('ntId', int), ('ruleId', int)])


class SymbolicSynthesizer(SolverPQ[PPTerm, Action]):
    def __init__(self, lib: FnLibrary, sort: PPFuncSort, nnprefix='', concreteTypes: List[PPSort] = []):
        self.lib = lib
        self.sort = sort
        self._ntNameGen = MiscUtils.getUniqueFn()
        self.nnprefix = nnprefix
        self.concreteTypes = concreteTypes

    def _giveUniqueNamesToUnks(self, st: PPTerm):
        def rename(nt: PPTermUnk):
            return PPTermUnk("nn_fun_%s_%d" % (self.nnprefix, self._ntNameGen()), nt.sort)

        return ASTUtils.applyTd(st, ASTUtils.isUnk, rename)

    def start(self) -> PPTerm:
        return PPTermNT('Z', self.sort)

    def setEvaluate(self, evaluate: bool):
        self.evaluate = evaluate

    def filterState(self, st: PPTerm):
        if ASTUtils.hasRedundantLambda(st):
            return False
        return True

    def getNextStates(self, st: PPTerm, action: Action) -> List[PPTerm]:
        rule = Rules.getRule(action.ruleId)
        nextSts = rule(self.lib, st, action.ntId)
        nextSts = list(filter(self.filterState, nextSts))
        return nextSts

    def getActionsAllNts(self, st: PPTerm) -> List[Action]:
        # This results in duplicate programs.
        numNts = ASTUtils.getNumNTs(st)
        actions = []
        for ntId in range(1, numNts + 1):
            for ruleId in range(Rules.numRules):
                actions.append(Action(ntId, ruleId))
        return actions

    def getActionsFirstNT(self, st: PPTerm) -> List[Action]:
        actions = []
        for ruleId in range(len(Rules.rules)):
            actions.append(Action(1, ruleId))
        return actions

    def getActions(self, st: PPTerm) -> List[Action]:
        return self.getActionsFirstNT(st)

    def getActionCost(self, st: PPTerm, action: Action) -> float:
        return ASTUtils.getSize(st)

    def isOpen(self, st: PPTerm) -> bool:
        return ASTUtils.isOpen(st)

    def hasUnk(self, st: PPTerm) -> bool:
        return ASTUtils.hasUnk(st)

    def onEachIteration(self, st: PPTerm, action: Action):
        return None
        print("------------")

        print(repr_py(st))
        print(st)
        print("Action: (%d, %d)" % (action.ntId, action.ruleId))

    def exit(self) -> bool:
        return False

    def genProgs(self) -> Iterable[Tuple[PPTerm, Dict[str, PPSort]]]:
        for prog in self.genTerms():
            if self.isOpen(prog):
                continue

            if self.concreteTypes:
                maxSortVarsToBeInstantiated = 2
                eprogs = instantiateSortVar(prog, self.concreteTypes, maxSortVarsToBeInstantiated)
            else:
                eprogs = [prog]

            for eprog in eprogs:
                unkSortMap = {}
                if self.hasUnk(eprog):
                    eprog = self._giveUniqueNamesToUnks(eprog)
                    unkSortMap = ASTUtils.getUnkNameSortMap(eprog)

                yield eprog, unkSortMap


def main():
    init_sort = PPFuncSort([PPInt()], PPBool())
    s = SymbolicSynthesizer(init_sort)
    s.solve()


if __name__ == '__main__':
    main()
