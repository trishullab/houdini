import logging
import time
from typing import Tuple, Dict, NamedTuple, List

from HOUDINI.Interpreter.Interpreter import Interpreter
from HOUDINI.InterpreterFilters import is_evaluable
from HOUDINI.FnLibraryFunctions import NotHandledException
from HOUDINI.FnLibrary import FnLibrary, PPLibItem
from HOUDINI.Synthesizer.AST import PPTerm, PPSort, PPFuncSort
from HOUDINI.Synthesizer.MiscUtils import getElapsedTime, formatTime
from HOUDINI.Synthesizer.ReprUtils import repr_py
from HOUDINI.Synthesizer.SymbolicSynthesizerEA import SymbolicSynthesizerEA

NeuralSynthesizerEASettings = NamedTuple("NeuralSynthesizerSettings", [
    ('G', int),  # Number of generations
    ('M', int),  # Evaluate at most M programs.
    ('K', int),  # Return top-k programs
])


class NSDebugInfo:
    def __init__(self, dprog):
        self.dprog = dprog


def _debug_info(prog: PPTerm, unkSortMap, lib: FnLibrary, fnSort: PPSort):
    lib_items = [PPLibItem(li.name, li.sort, None) for (_, li) in lib.items.items()]
    dprog = """
    io_examples_tr, io_examples_val = None, None
    prog = %s
    unkSortMap = %s
    lib = NewLibrary()
    lib.addItems(%s)
    fn_sort = %s
    interpreter = Interpreter(lib, epochs=1)
    res = interpreter.evaluate(program=prog,
                                        output_type_s=fn_sort.rtpe,
                                        unkSortMap=unkSortMap,
                                        io_examples_tr=io_examples_tr,
                                        io_examples_val=io_examples_val)
    """ % (str(prog), str(unkSortMap), str(lib_items), str(fnSort))

    return (NSDebugInfo(dprog),)


class NeuralSynthesizerEAResult:
    def __init__(self, top_k_solutions_results: List[Tuple[PPTerm, Dict]], progScores: List[List[Tuple[PPTerm,
                                                                                                       float]]],
                 numProgsEvaluated: int):
        # A list of top scoring programs
        self.top_k_solutions_results = top_k_solutions_results
        self.progScores = progScores # genid -> progid -> (PPTerm, float)
        self.numProgsEvaluated = numProgsEvaluated

    def get_top_solution_score(self):
        """
        Top scoring program and corresponding score
        """
        top_solution_score = self.top_k_solution_scores[0] if len(self.top_k_solution_scores) else None
        return top_solution_score


# region Logging Utilities
def logEvaluatedProgram(prog):
    print("Program evaluated: %s" % repr_py(prog))


def logRejectedProgram(prog, ecode):
    print("Program rejected (ecode %d): %s" % (ecode, repr_py(prog)))
    print("Program rejected (pyrep): %s" % str(prog))


def logUnhandledProgram(prog):
    print("Program not handled: %s" % repr_py(prog))


def logEvaluatorException(e, prog, unkSortMap, lib, sort):
    loggerE = logging.getLogger('pp.exceptions')
    loggerE.error('Exception in the Interpreter.\n %s' % repr(e))
    loggerE.error('DebugInfo.\n %s' % _debug_info(prog, unkSortMap, lib, sort)[0].dprog)


def logIsEvaluableException(e, prog, unkSortMap, lib, sort):
    loggerE = logging.getLogger('pp.exceptions')
    e.args += _debug_info(prog, unkSortMap, lib, sort)
    loggerE.error('#### Exception in the Interpreter.\n %s' % repr(e))
# endregion


class NeuralSynthesizerEA:
    def __init__(self, interpreter: Interpreter, synthesizer: SymbolicSynthesizerEA,
                 lib: FnLibrary, sort: PPFuncSort, settings: NeuralSynthesizerEASettings):
        self.interpreter = interpreter
        self.synthesizer = synthesizer
        self.lib = lib
        self.sort = sort
        self.settings = settings
        self.prog_unkinfo_tuples = []  # TODO: Remove this

    def interpret(self, prog, unkSortMap, ioExamplesTr, ioExamplesVal, ioExamplesTest) -> Dict:
        output_type = self.sort.rtpe
        print('BEGIN_EVALUATE, Time: %s' % getElapsedTime())
        eStart = time.time()
        res = self.interpreter.evaluate(program=prog,
                                        output_type_s=output_type,
                                        unkSortMap=unkSortMap,
                                        io_examples_tr=ioExamplesTr,
                                        io_examples_val=ioExamplesVal,
                                        io_examples_test=ioExamplesTest)
        print('END_EVALUATE, Time: %s' % getElapsedTime())
        eEnd = time.time()
        print("TIME_TAKEN_EVALUATE, %s" % formatTime(eEnd - eStart))

        return res

    def updateTopK(self, topKSolutionResults: List[Tuple[PPTerm, Dict]]):
        topKSolutionResults.sort(key=lambda x: x[1]['accuracy'], reverse=True)
        if len(topKSolutionResults) > self.settings.K:
            del topKSolutionResults[-1]

        for i in range(1, topKSolutionResults.__len__()):
            topKSolutionResults[i][1]["new_fns_dict"] = None


    def interpret2(self, prog, unkSortMap, ioExamplesTr, ioExamplesVal, ioExamplesTest):
        try:
            interpreterRes = self.interpret(prog, unkSortMap, ioExamplesTr, ioExamplesVal, ioExamplesTest)
        except NotHandledException as e:
            logUnhandledProgram(prog)
            interpreterRes = None
        # except Exception as e: # TODO: Enable this
        #     e.args += _debug_info(prog, unkSortMap, self.lib, self.sort)
        #     traceback.print_exc()
        #     logEvaluatorException(e, prog, unkSortMap, self.lib, self.sort)
        #     interpreterRes = None

        return interpreterRes

    def isEvaluable2(self, prog, unkSortMap, ns):
        try:
            isOk, ecode = is_evaluable(prog, ns)
        except Exception as e:
            raise e
            logIsEvaluableException(e, prog, unkSortMap, self.lib, self.sort)
            isOk = False
            ecode = -1

        if ecode > 4:
            logRejectedProgram(prog, ecode)

        return isOk, ecode

    def solve(self, ioExamplesTr, ioExamplesVal, ioExamplesTest) -> List[Tuple[PPTerm, float]]:
        topKSolutionResults = []
        m = 0
        progScoresAll = []

        def processProg(prog, unkSortMap):
            # isOk, _ecode = self.isEvaluable2(prog, unkSortMap, self.sort.rtpe)
            isOk, _ecode = self.isEvaluable2(prog, unkSortMap, self)
            if isOk:
                nonlocal m
                m += 1
                interpreterRes = self.interpret2(prog, unkSortMap, ioExamplesTr, ioExamplesVal, ioExamplesTest)
                if interpreterRes is not None:
                    progScore = interpreterRes['accuracy']
                    topKSolutionResults.append((prog, interpreterRes))
                    self.updateTopK(topKSolutionResults)
                else:
                    progScore = - 100.0  # - float('int) is causing problem in randomProportionalSelection
            else:
                progScore = - 100.0

            return progScore

        def processGen(progUnkList):
            progScoresGen = []
            for prog, unkSortMap in progUnkList:
                assert prog is not None
                score = processProg(prog, unkSortMap)
                progScoresGen.append((prog, score))
                if m >= self.settings.M:
                    break
            return progScoresGen

        def processGens():
            progScores = None
            for genId in range(self.settings.G):
                progUnkList = self.synthesizer.genProgs(progScores)
                progScores = processGen(progUnkList)
                progScoresAll.append(progScores)
                if m >= self.settings.M:
                    break

        processGens()
        print('Generations: %d' % len(progScoresAll))  #TODO: Remove this
        return NeuralSynthesizerEAResult(topKSolutionResults, progScoresAll, m)
