import os
import pickle
from typing import NamedTuple, List, Dict

from HOUDINI.Eval.EvaluatorUtils import mk_tag, write_to_file, append_to_file
# from HOUDINI.Eval.EvaluatorTask import Task, TaskResult, TaskResultSingle
from HOUDINI.Eval.Task import Task, TaskResult, TaskResultSingle
from HOUDINI.FnLibraryFunctions import loadLibrary1
from HOUDINI.FnLibrary import FnLibrary, PPLibItem
from HOUDINI.Synthesizer import ASTUtils
from HOUDINI.Synthesizer.AST import PPSort
from HOUDINI.Synthesizer.GenUtils import createDir

# from HOUDINI.Eval.EvaluatorTaskSeq import TaskSeqSettings

TaskSeqSettings = NamedTuple('TaskSeqSettings', [
    ('update_library', bool),  # Update the library after each task
    ('results_dir', str),  # Results directory
])
# import sys
# import psutil


class TaskSeq:
    def __init__(self, tasks: List[Task], seq_settings, lib):
        self.seq_settings = seq_settings
        self.tasks = tasks
        self.lib = lib

    def name(self):
        raise NotImplementedError()

    def sname(self):
        raise NotImplementedError()

    def getLibLocation(self):
        return '%s/%s/Lib' % (self.seq_settings.results_dir, self.name())

    def get_seq_dir(self):
        results_dir = self.seq_settings.results_dir
        directory = results_dir + '/' + self.name()
        return directory

    def get_pickle_path(self, task_name):
        directory = self.get_seq_dir()
        file_path = directory + '/' + task_name + '.pickle'
        return file_path

    def save_results(self, task: Task, task_result: TaskResult):
        file_path = self.get_pickle_path(task.name())
        seq_dir = self.get_seq_dir()

        createDir(seq_dir)

        with open(file_path, 'wb') as fh:
            pickle.dump(task_result, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def update_library(self, lib: FnLibrary, task_result_single: TaskResultSingle, taskid):
        if self.seq_settings.update_library:
            # Add learned modules to the library
            top_solution = task_result_single.get_top_solution_details()
            if top_solution is not None:
                prog, resDict = top_solution
                unk_sort_map: Dict[str, PPSort] = ASTUtils.getUnkNameSortMap(prog)
                lib_items = [PPLibItem(unk, unk_sort, resDict['new_fns_dict'][unk]) for unk, unk_sort in
                             unk_sort_map.items()]
                if lib_items.__len__() > 0:
                    lib.addItems(lib_items)
                # Save the library.
                lib.save1(self.getLibLocation(), taskid)

    """
    @staticmethod
    def cpuStats(): #useful for debugging memory usage
        print(sys.version)
        print(psutil.cpu_percent())
        print(psutil.virtual_memory())  # physical memory usage
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
        print('memory GB:', memoryUse)
    """

    def run(self, id) -> TaskResult:
        task = self.tasks[id]

        # Load the library from disk
        print("Loading Library...")
        pickle_path = self.get_pickle_path(task.name())
        if os.path.exists(pickle_path):
            os.remove(pickle_path)
            # raise Exception("Result file already exists: %s" % pickle_path)

        lib = loadLibrary1(self.getLibLocation(), id) if id > 0 else None
        if lib is not None:
            print("Library loaded")
            self.lib = lib
        else:
            print("Library not loaded. (May not be available)")

        task_result = task.run()

        # print("FINISHED THE TASK. HERE's the cpu status")
        # TaskSeq.cpuStats()

        print("Saving Library")
        last_task_result = task_result.results[-1]
        self.update_library(task.seq.lib, last_task_result, id)

        print("Appending to the report")
        # self.save_results(task, task_result)
        self.append_to_the_report(id, task_result)

    def append_to_the_report(self, task_id, task_result):
        seq_dir = self.get_seq_dir()
        report_file_path = seq_dir + '/' + self.name() + '.html'

        if task_id == 0:
            header = mk_tag('h1', self.name())
            write_to_file(report_file_path, header)

        task = self.tasks[task_id]

        task_result.save_plot(task, seq_dir)
        report = task_result.gen_report(task, task_id)
        append_to_file(report_file_path, report)

    """"""
    def write_report(self, task_id):
        """write report upto task_id """
        seq_dir = self.get_seq_dir()
        report_file_path = seq_dir + '/' + self.name() + '.html'
        header = mk_tag('h1', self.name())
        write_to_file(report_file_path, header)

        for tid in range(task_id + 1):
            task = self.tasks[tid]
            task_result = None
            pickle_file_path = self.get_pickle_path(task.name())
            # print('PATHS #########')
            # print(getPath())
            # print(getPythonPath())
            with open(pickle_file_path, 'rb') as fh:
                task_result = pickle.load(fh)

            task_result.save_plot(task, seq_dir)
            report = task_result.gen_report(task, tid + 1)
            append_to_file(report_file_path, report)
