import random
from functools import wraps
from itertools import product
from typing import List, Optional, Tuple, Iterable, Dict

from HOUDINI.FnLibraryFunctions import get_items_from_repo
from HOUDINI.FnLibrary import FnLibrary
from HOUDINI.Synthesizer import MiscUtils, Rules, ASTUtils
from HOUDINI.Synthesizer.AST import PPTerm, PPTermNT, PPFuncSort, PPTermUnk, PPSort, PPFuncApp, PPDimVar
from HOUDINI.Synthesizer.ASTDSL import mkFuncSort, mkRealTensorSort, mkBoolTensorSort
from HOUDINI.Synthesizer.ASTUtils import progTreeSize, inferType, \
    deconstructProg, constructProg, applyTdProgGeneral, isAbstract, progDepth
from HOUDINI.Synthesizer.IntermediateTypeHandler import instantiateSortVar
from HOUDINI.Synthesizer.MiscUtils import logEntryExit
from HOUDINI.Synthesizer.ReprUtils import repr_py, repr_py_sort, repr_py_ann
from HOUDINI.Synthesizer.SymbolicSynthesizer import Action

POPULATION_SIZE = 20

class ProgramGenerator:
    def __init__(self, lib: FnLibrary):
        # self.d = maxProgramDepth
        self.lib = lib

    def getActionsFirstNT(self, st: PPTerm) -> List[Action]:
        return [Action(1, ruleId) for ruleId, rule in enumerate(Rules.rules)]

    def getNextRandomSt(self, state):
        actions = self.getActionsFirstNT(state)
        random.shuffle(actions)
        for action in actions:
            rule = Rules.getRule(action.ruleId)
            try:
                nextSts = rule(self.lib, state, action.ntId)
            except SystemError:
                print('SystemError while rule execution')
                nextSts = []

            nextSts = [s for s in nextSts if s is not None]
            if nextSts:
                nextSt = random.choice(nextSts)
                return nextSt

        return None

    def genProgDeco(method):
        @wraps(method)
        def _impl(self, sort):
            print('begin genProg')
            # print(sort)
            print(repr_py_sort(sort))
            program = method(self, sort)
            if program is not None:
                # print(program)
                print(repr_py(program))
            else:
                print('None')
            print('end genProg')
            return program

        return _impl

    # @genProgDeco
    def genProg(self, sort: PPSort, maxDepth: int) -> Optional[PPTerm]:
        state = PPTermNT('Z', sort)
        di = 0

        while di < maxDepth:
            nxtState = self.getNextRandomSt(state)
            if nxtState is None:
                return None
            elif not ASTUtils.isOpen(nxtState):
                return nxtState
            else:  # isOpen
                state = nxtState
            di += 1

        return None


class SymbolicSynthesizerEA:
    def __init__(self, lib: FnLibrary, sort: PPFuncSort, nnprefix='', concreteTypes: List[PPSort]=[]):
        self.size = POPULATION_SIZE  # population size
        self.lib = lib
        self.sort = sort
        self._ntNameGen = MiscUtils.getUniqueFn()
        # self.curGen: List[PPTerm] = None
        self.programGenerator = ProgramGenerator(self.lib)
        self.nnprefix = nnprefix
        self.concreteTypes = concreteTypes

    def _giveUniqueNamesToUnks(self, st: PPTerm):
        def rename(nt: PPTermUnk):
            if nt.name == 'Unk':
                return PPTermUnk("nn_fun_%s_%d" % (self.nnprefix, self._ntNameGen()), nt.sort)
            else:  # Renaming not required.
                return nt

        return ASTUtils.applyTd(st, ASTUtils.isUnk, rename)

    def initGen(self):
        gen = []

        while len(gen) < self.size:
            prog = self.programGenerator.genProg(self.sort, 6)
            if prog is not None and prog not in gen:
                gen.append(prog)

        return gen

    # @logEntryExit('SymbolicSynthesizerEA::Select')
    def select(self, progScores: List[Tuple[PPTerm, float]]) -> List[PPTerm]:
        newGen = []
        # add 0.01 to all scores to avoid division by zero
        scores = [s for p, s in progScores]
        for _ in range(len(scores)):
            index = randomProportionalSelection(scores)
            newGen.append(progScores[index][0])

        return newGen

    def crossoverPopulation(self, gen: List[PPTerm]):
        newGen = []
        random.shuffle(gen)
        halfsize = self.size // 2
        for i in range(halfsize):
            c1, c2 = crossover(gen[2 * i], gen[2 * i + 1], self.lib)
            newGen.append(c1)
            newGen.append(c2)

        return newGen

    def mutatePopulation(self, gen: List[PPTerm]):
        return [mutate(p, self.lib) for p in gen]

    def instantiateAbstractTypes(self, progs):
        newProgs = []
        for prog in progs:
            if self.concreteTypes:
                maxSortVarsToBeInstantiated = 2
                eprogs = instantiateSortVar(prog, self.concreteTypes, maxSortVarsToBeInstantiated)
                newProgs.extend(eprogs)
            else:
                newProgs.append(prog)

        # Drop extra programs
        # random.shuffle(newProgs)
        # return newProgs[:len(progs)]
        return newProgs

    def evolve(self, progScores: List[Tuple[PPTerm, float]]):
        res = None
        if progScores is None:
            res = self.initGen()
        else:
            genv1 = self.select(progScores)
            genv2 = self.crossoverPopulation(genv1)
            genv3 = self.mutatePopulation(genv2)
            # oldProgs = [p for p, s in progScores]
            # genv4 = self.retainTopK(oldProgs, genv3)
            res = genv3

        res = self.instantiateAbstractTypes(res)
        return res

    def logGenProgs(_genProgs):
        @wraps(_genProgs)
        def impl(self, scores):
            print('GENPROGS START: scores: ', str(scores))
            progUnks = _genProgs(self, scores)
            print('GENPROGS END: progs: ', ['None' if p is None else repr_py(p) for (p, unks) in progUnks])
            # print('progs: ', progUnks)
            return progUnks

        return impl

    def getProgsAndUnkSortWithUniqueNames(self, progs):
        res = []
        for prog in progs:
            assert (not ASTUtils.isOpen(prog))

            unkSortMap = {}
            if ASTUtils.hasUnk(prog):
                prog = self._giveUniqueNamesToUnks(prog)
                unkSortMap = ASTUtils.getUnkNameSortMap(prog)

            res.append((prog, unkSortMap))

        return res


    @logEntryExit('SymbolicSynthesizerEA::genProgs')
    def genProgs(self, progScores: List[Tuple[PPTerm, float]]) -> Iterable[Tuple[PPTerm, Dict[str, PPSort]]]:
        programs = self.evolve(progScores)
        progUnkScores = self.getProgsAndUnkSortWithUniqueNames(programs)
        return progUnkScores


def getChildSorts(term: PPFuncApp, lib):
    fnName = term.fn.name.replace('lib.', '')
    li = lib.get(fnName)
    fnSort = li.sort
    childSorts = fnSort.args
    return childSorts


def logMutate(afn):
    @wraps(afn)
    def impl(prog, lib):
        try:
            res = afn(prog, lib)
            print('')
            print('Mutate Input :', repr_py_ann(prog) if prog else 'None')
            print('Mutate Output:', repr_py_ann(res) if res else 'None')
            return res
        except Exception as e:
            print('Prog', prog)
            print('Res', res)
            raise e
    return impl


# @logMutate
def mutate(prog: PPTerm, lib: FnLibrary) -> PPTerm:
    size = progTreeSize(prog)
    rid = random.randrange(0, size)

    pg = ProgramGenerator(lib)

    cid = -1

    def mutate_rec(term):
        nonlocal cid
        cid += 1
        newTerm = None
        if cid < rid:
            if isinstance(term, PPFuncApp):
                childs = deconstructProg(term)
                childSorts = getChildSorts(term, lib)
                newChilds = []
                for c, cs in zip(childs, childSorts):
                    if isinstance(cs, PPDimVar):
                        # Do not mutate DimVars that are direct function params. Example: zeros
                        nc = c
                    else:
                        nc = mutate_rec(c)
                    newChilds.append(nc)
                newTerm = constructProg(term, newChilds)
            else:
                childs = deconstructProg(term)
                newChilds = [mutate_rec(c) for c in childs]
                newTerm = constructProg(term, newChilds)
        elif cid == rid:
            try:
                tpe = inferType(term, lib)
                if tpe is not None:
                    N = 10
                    for i in range(N):
                        maxDepth = progDepth(term) + 2
                        newTerm = pg.genProg(tpe, maxDepth)
                        if newTerm is not None:
                            break

                # return same term if not able to generate a random term after N attempts.
                if newTerm is None:
                    newTerm = term
            except ValueError as e:
                print('ValueError: %s' % e)
                print('term: %s' % str(term))
                newTerm = term

        elif cid > rid:
            newTerm = term

        return newTerm

    newProg = mutate_rec(prog)
    return newProg


def getIdTermSorts(prog, lib):
    """
    Some IDs might be missing if inferType returns None.
    """
    idTermSorts = []
    cid = -1

    def func(term):
        nonlocal cid
        cid += 1
        try:
            sort = inferType(term, lib)
        except ValueError as e:
            print('ValueError: %s' % e)
            print('prog: %s' % str(prog))
            sort = None

        if sort is not None:
            idTermSorts.append((cid, term, sort))
        return None

    applyTdProgGeneral(prog, func)

    return idTermSorts


def getSubTerm(prog, tid):
    subTerm = None
    cid = -1

    def func(term):
        nonlocal cid
        cid += 1

        if cid < tid:
            # continue with the search
            return None
        elif cid == tid:
            nonlocal subTerm
            subTerm = term
            return term
        elif cid > tid:
            # stop searching
            return term

    applyTdProgGeneral(prog, func)
    return subTerm


def replaceSubTerm(prog, tid, newTerm):
    cid = -1

    def func(term):
        nonlocal cid
        cid += 1

        if cid < tid:
            # continue with the search
            return None
        elif cid == tid:
            return newTerm
        elif cid > tid:
            # stop searching
            return term

    newProg = applyTdProgGeneral(prog, func)
    return newProg


def getCompatibleIdPairs(idTermSorts1, idTermSorts2):
    pairs = []
    for (id1, t1, s1), (id2, t2, s2) in product(idTermSorts1, idTermSorts2):
        if s1 == s2 \
                and t1 != t2 \
                and not (id1 == 0 and id2 == 0) \
                and not isAbstract(s1):
            pairs.append(((id1, t1, s1), (id2, t2, s2)))
    return pairs


def crossover(term1, term2, lib, selPairIndex=None):
    if term1 == term2:
        return term1, term2

    idTermSorts1 = getIdTermSorts(term1, lib)
    idTermSorts2 = getIdTermSorts(term2, lib)

    idTermSortPairs = getCompatibleIdPairs(idTermSorts1, idTermSorts2)

    if not idTermSortPairs:
        return term1, term2

    if selPairIndex is not None:
        (id1, t1, s1), (id2, t2, s2) = idTermSortPairs[selPairIndex]
    else:
        (id1, t1, s1), (id2, t2, s2) = random.choice(idTermSortPairs)

    c1 = replaceSubTerm(term1, id1, t2)
    c2 = replaceSubTerm(term2, id2, t1)

    return c1, c2


def mkDefaultLib() -> FnLibrary:
    lib = FnLibrary()
    lib.addItems(get_items_from_repo(['compose', 'repeat', 'map_l', 'fold_l', 'conv_l', 'zeros']))
    return lib


def main():
    input_type = mkRealTensorSort([1, 1, 28, 28])
    output_type = mkBoolTensorSort([1, 1])
    fn_sort = mkFuncSort(input_type, output_type)
    lib = mkDefaultLib()
    pg = ProgramGenerator(lib)
    progs = []
    while len(progs) < 30:
        prog = pg.genProg(fn_sort)
        if prog is not None and prog not in progs:
            print(repr_py(prog))
            progs.append(prog)
        else:
            pass
            # print(None)


# @logEntryExit('SymbolicSynthesizerEA::randomProportionalSelection')
def randomProportionalSelection(freqs):
    # make them all positive
    minFreq = min(freqs)
    if minFreq < 0.0:
        freqs = [f - minFreq for f in freqs]

    # Ensure that maxFreq is not zero
    maxFreq = max(freqs)
    if maxFreq < 0.00001:
        freqs = [f + 0.00001 for f in freqs]

    maxFreq = max(freqs)

    n = len(freqs)
    while True:
        i = int(n * random.random())
        if random.random() < (freqs[i] / float(maxFreq)):
            return i


def main2():
    # freqs = [0.1, 0.2, 0.4, 0.8, 0.16, 0.5, 0.10]
    freqs = [-100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -4.3]
    # freqs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    max_freq = max(freqs)
    # N = sum(freqs) * 100000
    N = 100000
    i = 0
    newFreqs = [0] * len(freqs)
    while i < N:
        idx = randomProportionalSelection(freqs)
        newFreqs[idx] += 1
        i += 1

    print(newFreqs)


if __name__ == '__main__':
    main2()
