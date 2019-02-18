from queue import PriorityQueue
from typing import List, TypeVar, Generic, Optional, Tuple, Iterable

from HOUDINI.Synthesizer import MiscUtils, ReprUtils
from HOUDINI.Synthesizer.AST import PPTerm

S = TypeVar('S')
A = TypeVar('A')


class SolverPQ(Generic[S, A]):
    def start(self) -> S:
        pass

    def getNextStates(self, st: S, action: A) -> List[S]:
        pass

    def getActions(self, st: S) -> List[A]:
        pass

    def getActionCost(self, st: S, action: A) -> float:
        pass

    def isOpen(self, st: S) -> bool:
        pass

    def onEachIteration(self, st: S, action: A):
        pass

    def getScore(self, st: S) -> float:
        pass

    def exit(self) -> bool:
        pass

    def isEvaluable(self) -> bool:
        pass

    def genTerms(self) -> Iterable[S]:

        sn = MiscUtils.getUniqueFn()
        pq = PriorityQueue()

        def addToPQ(aState):
            for cAction in self.getActions(aState):
                stateActionScore = self.getActionCost(aState, cAction)
                pq.put((stateActionScore, sn(), (aState, cAction)))

        solution, score = None, 0

        state = self.start()
        addToPQ(state)

        while not pq.empty() and not self.exit():
            _, _, (state, action) = pq.get()

            self.onEachIteration(state, action)

            states = self.getNextStates(state, action)

            for state in states:
                if self.isOpen(state):
                    addToPQ(state)
                yield state
