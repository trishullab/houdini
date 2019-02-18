import matplotlib
import time

from HOUDINI.Synthesizer.MiscUtils import getElapsedTime, formatTime
from HOUDINI.Synthesizer.ASTDSL import mkRealTensorSort, mkBoolTensorSort

matplotlib.use('agg')
import matplotlib.pyplot as plt
from HOUDINI.Synthesizer.ReprUtils import repr_py  # added for debugging purposes

from typing import NamedTuple, List, Tuple, Dict

import numpy as np
import os
from HOUDINI.Eval.EvaluatorUtils import iterate_diff_training_sizes, mk_tag, mk_div
from HOUDINI.Interpreter.Interpreter import Interpreter
from HOUDINI.NeuralSynthesizer import NeuralSynthesizerSettings, NeuralSynthesizer, NeuralSynthesizerResult, \
    NSDebugInfo
from HOUDINI.NeuralSynthesizerEA import NeuralSynthesizerEA, NeuralSynthesizerEASettings, \
    NeuralSynthesizerEAResult
from HOUDINI.Synthesizer.SymbolicSynthesizerEA import SymbolicSynthesizerEA

from HOUDINI.Synthesizer.AST import PPSort, PPTerm
from HOUDINI.Synthesizer.ReprUtils import repr_py
from HOUDINI.Synthesizer.SymbolicSynthesizer import SymbolicSynthesizer


class TaskResultSingle:
    """
    Task Result for a single datasize
    """
    def __init__(self):
        self.top_k_solutions_results: List[Tuple[PPTerm, Dict]] = []  # top k solution-score pairs. descending order
        self.num_programs: int = None  # Total number of programs evaluated
        self.time: int = None  # time taken in seconds
        self.progScores: List[List[Tuple[str, float]]] = None  # genid --> progid --> (progstr, score)
        # Only for evolutionary synthesizer. List of programs for various generations.

    def get_top_solution_details(self) -> Tuple[PPTerm, Dict]:
        return self.top_k_solutions_results[0] if len(self.top_k_solutions_results) > 0 else None

    def get_top_program(self) -> PPTerm:
        prog_res_dict = self.get_top_solution_details()
        if prog_res_dict is not None:
            return prog_res_dict[0]
        else:
            return None

    def get_top_score(self) -> float:
        prog_res_dict = self.get_top_solution_details()
        if prog_res_dict is not None:
            return prog_res_dict[1]['test_accuracy']
        else:
            return None

    def gen_report(self, task) -> str:
        header_row = mk_tag('tr', mk_tag('th', 'Score Test / Val') + mk_tag('th', 'Top %d Programs' % task.settings.K))

        def mk_row(p, s_test, s_val):
            return mk_tag('tr', mk_tag('td',"%.4f / %.4f" % (s_test, s_val)) + mk_tag('td', repr_py(p)))

        rows = []
        for p, rdict in self.top_k_solutions_results:
            test_acc = rdict['test_accuracy'] if 'test_accuracy' in rdict else -10000.
            val_acc = rdict['accuracy'] if 'accuracy' in rdict else -10000.
            rows.append(mk_row(p, test_acc, val_acc))
        # rows = [mk_row(p, rdict['test_accuracy'], rdict['accuracy']) for p, rdict in self.top_k_solutions_results]

        table_content = '\n'.join([header_row] + rows)

        res = mk_tag('table', table_content, attribs={'border': '1'})

        def gen_prog_content():
            genReprs = []

            for i, iprogs in enumerate(self.progScores):
                c = mk_div('Generation: %d' % i)
                c += ''.join(mk_div('%.2f ' % score + pstr, cls=['Prog']) for (pstr, score) in iprogs)
                genReprs.append(mk_div(c, cls=['Generation']))

            return ''.join(genReprs)

        if self.progScores:
            prog_content = gen_prog_content()
            res = res + mk_div(prog_content, cls=['Programs'])

        return res

    def get_raw_data(self):
        res = dict()
        res['top_k_solutions_results'] = [(str(prog), res)for (prog, res) in self.top_k_solutions_results]
        res['num_programs'] = self.num_programs
        res['time'] = self.time
        res['progScores'] = self.progScores
        return res


class TaskResult:
    def __init__(self):
        self.results: List[TaskResultSingle] = []  # Result for various data sizes.

    def save_plot(self, task, seq_dir):
        xs = np.array(task.settings.training_percentages)
        ys = np.array([single_result.get_top_score() for single_result in self.results])

        plt.figure()
        plt.xlabel("Training Dataset Size")
        plt.ylabel("Accuracy")
        handles = []
        t_line = plt.plot(
            xs,
            ys,
            label=task.name(),
            marker='o')
        handles.append(t_line)

        # plt.legend(handles=handles)
        # plt.show(block=True)
        img_path = seq_dir + '/' + task.name() + '.png'
        plt.savefig(img_path)

        xydata = np.array([xs, ys])
        np.save(seq_dir + '/' + task.name() + '_plot.npy', xydata)
        plt.close()

    def gen_report(self, task: 'Task', tid: int) -> str:
        res = mk_tag('h2', 'Task %d: %s' % (tid, task.name()))
        # data size - score plot
        img_file = task.name() + '.png'
        res += mk_div(mk_tag('img', '', attribs={'src': img_file}))

        # Top k Programs and scores
        for task_result_single, percentage in zip(self.results, task.settings.training_percentages):
            res += '<br>'
            res += mk_div('Training Data Used: %s %%' % percentage)
            res += mk_div('Number of programs evaluated: %s' % task_result_single.num_programs)
            res += task_result_single.gen_report(task)
        return res

    def get_raw_data(self):
        return [result.get_raw_data() for result in self.results]


_TaskSettings = NamedTuple('TaskSettings', [
    ('train_size', int),  # Training data size
    ('val_size', int),  # Validation data size = Test data size
    ('training_percentages', List[int]),  # Train on different training data sizes.
    ('N', int),  # Max number of solutions to generate
    ('M', int),  # Max number of solutions evaluated by the interpreter
    ('K', int),  # Number of top solutions to store.
    ('epochs', int),
    ('synthesizer', str),  # 'enumerative'| 'evolutionary',
    ('batch_size', int),
    ('dbg_learn_parameters', bool)  # If False, it won't learn new parameters
])


class TaskSettings(_TaskSettings):
    """
    A named tuple with default parameters
    """
    def __new__(cls, batch_size=150, dbg_learn_parameters=True, **kwargs):
        return super(TaskSettings, cls).__new__(cls, batch_size=batch_size, dbg_learn_parameters=dbg_learn_parameters,
                                                **kwargs)


class Task:
    def __init__(self, fn_sort: PPSort, settings: TaskSettings, seq: 'TaskSeq', dbg_learn_parameters=True):
        self.fn_sort = fn_sort
        self.settings = settings
        self.seq = seq

        self.dbg_learn_parameters = dbg_learn_parameters

    def name(self):
        return NotImplementedError()

    def sname(self):
        return NotImplementedError()

    def _mkNSynth(self):
        ea_synthesis_mode = self.settings.synthesizer == 'evolutionary'
        interpreter = Interpreter(self.seq.lib, epochs=self.settings.epochs, batch_size=self.settings.batch_size)
        nnprefix = self.seq.sname() + self.sname()

        if self.settings.synthesizer == 'enumerative':
            # concrete_types = [mkRealTensorSort([1, 64, 4, 4]), mkRealTensorSort([1, 50])]
            concreteTypes = [mkRealTensorSort([1, 64, 4, 4]), mkBoolTensorSort([1, 1]), mkRealTensorSort([1, 50])]
            synth = SymbolicSynthesizer(self.seq.lib, self.fn_sort, nnprefix, concreteTypes)

            ns_settings = NeuralSynthesizerSettings(self.settings.N, self.settings.M, self.settings.K)
            assert self.seq.lib is not None
            nsynth = NeuralSynthesizer(interpreter, synth, self.seq.lib, self.fn_sort, self.settings.dbg_learn_parameters, ns_settings)
            return nsynth
        elif self.settings.synthesizer == 'evolutionary':
            concreteTypes = [mkRealTensorSort([1, 64, 4, 4]), mkBoolTensorSort([1, 1]), mkRealTensorSort([1, 50])]
            synth = SymbolicSynthesizerEA(self.seq.lib, self.fn_sort, nnprefix, concreteTypes)

            # TODO: Do not hardcode G
            NUM_GENERATIIONS = 100
            ns_settings = NeuralSynthesizerEASettings(G=NUM_GENERATIIONS, M=self.settings.M, K=self.settings.K)
            assert self.seq.lib is not None
            nsynth = NeuralSynthesizerEA(interpreter, synth, self.seq.lib, self.fn_sort, ns_settings)
            return nsynth

    def run(self) -> TaskResult:
        tStart = time.time()
        print("BEGIN_TASK, Time: %s" % getElapsedTime())
        nsynth = self._mkNSynth()

        print("Num of programs selected for evaluation: %d" % len(nsynth.prog_unkinfo_tuples))
        print("Programs selected for evaluation:")
        for c_prog, c_unkSortMap in nsynth.prog_unkinfo_tuples:
            print(repr_py(c_prog))
            print(c_unkSortMap)

        train_io, val_io, test_io = self.get_io_examples()

        max_iterations = \
            (self.settings.train_size // nsynth.interpreter.batch_size + (
                1 if self.settings.train_size % nsynth.interpreter.batch_size != 0 else 0)) * self.settings.epochs

        res = TaskResult()
        for i, (c_tr_io_examples, c_tr_num_items) in enumerate(iterate_diff_training_sizes(train_io,
                                                                              self.settings.training_percentages)):
            interpreter = nsynth.interpreter
            c_iterations_per_epoch = c_tr_num_items // interpreter.batch_size + (1 if c_tr_num_items % interpreter.batch_size != 0 else 0)
            c_num_epochs = max_iterations // c_iterations_per_epoch + (1 if max_iterations % c_iterations_per_epoch != 0 else 0)
            # interpreter.epochs = c_num_epochs

            rStart = time.time()
            print("BEGIN_RUN %d, Time: %s" % (i, getElapsedTime()))
            c_res = TaskResultSingle()

            try:
                nsynth_res: NeuralSynthesizerResult = nsynth.solve(c_tr_io_examples, val_io, test_io)

                c_res.top_k_solutions_results = nsynth_res.top_k_solutions_results
                c_res.num_programs = len(nsynth.prog_unkinfo_tuples)
                c_res.time = None  # TODO: implement

                if isinstance(nsynth_res, NeuralSynthesizerEAResult):
                    c_res.num_programs = nsynth_res.numProgsEvaluated
                    c_res.progScores = [[(repr_py(p), s) for p, s in gProgScores]
                                        for gProgScores in nsynth_res.progScores]

                dir_path = self.seq.get_seq_dir()
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)

                # top_result_evaluations_np = c_res.top_k_solutions_results[0][1]["evaluations_np"]
                # np.save("{}/__{}_{}.npy".format(self.seq.get_seq_dir(), nsynth.synthesizer.nnprefix, c_tr_num_items), top_result_evaluations_np)

            except Exception as e:
                print("Exception in NeuralSynthesizer.solve: %s" % str(e))
                print("# Task Name: %s" % self.name())
                for a in e.args:
                    if isinstance(a, NSDebugInfo):
                        print(a.dprog)
                raise

            res.results.append(c_res)
            print("END_RUN %d, Time: %s" % (i, getElapsedTime()))
            rEnd = time.time()
            print("TIME_TAKEN_RUN, %s" % formatTime(rEnd - rStart))

        print("END_TASK, Time: %s" % getElapsedTime())
        tEnd = time.time()
        print("TIME_TAKEN_TASK, %s" % formatTime(tEnd - tStart))

        if hasattr(nsynth, "target_program_position"):
            print("Program found at place : {}".format(nsynth.target_program_position))

        return res
