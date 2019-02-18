from HOUDINI.Eval.EvaluatorUtils import get_io_examples_recognize_digit, \
    get_io_examples_count_digit_occ, get_io_examples_count_toys, get_io_examples_recognize_toy
from HOUDINI.Eval.Task import Task
from HOUDINI.Synthesizer.ASTDSL import mkFuncSort, mkRealTensorSort, mkListSort, mkBoolTensorSort


class RecognizeDigitTask(Task):
    def __init__(self, settings, digit, seq, dbg_learn_parameters):
        input_type = mkRealTensorSort([1, 1, 28, 28])
        output_type = mkBoolTensorSort([1, 1])
        fn_sort = mkFuncSort(input_type, output_type)

        super(RecognizeDigitTask, self).__init__(fn_sort, settings, seq, dbg_learn_parameters)
        self.digit = digit

    def get_io_examples(self):
        return get_io_examples_recognize_digit(self.digit,
                                               self.settings.train_size,
                                               self.settings.val_size)

    def name(self):
        return "recognize_digit_%s" % self.digit

    def sname(self):
        return "r%d" % self.digit


class RecognizeToyTask(Task):
    def __init__(self, settings, toy, seq, dbg_learn_parameters):
        input_type = mkRealTensorSort([1, 1, 28, 28])
        output_type = mkBoolTensorSort([1, 1])
        fn_sort = mkFuncSort(input_type, output_type)
        self.toy = toy

        super(RecognizeToyTask, self).__init__(fn_sort, settings, seq, dbg_learn_parameters)

    def get_io_examples(self):
        return get_io_examples_recognize_toy(self.toy, self.settings.train_size, self.settings.val_size)

    def name(self):
        return "recognize_toy"

    def sname(self):
        return "rt"


class CountToysTask(Task):
    def __init__(self, settings, toy_class, seq, dbg_learn_parameters):
        input_type = mkListSort(mkRealTensorSort([1, 1, 28, 28]))
        output_type = mkRealTensorSort([1, 1])
        fn_sort = mkFuncSort(input_type, output_type)

        super(CountToysTask, self).__init__(fn_sort, settings, seq, dbg_learn_parameters)
        self.toy_class = toy_class

    def get_io_examples(self):
        return get_io_examples_count_toys(self.toy_class, self.settings.train_size, self.settings.val_size)

    def name(self):
        return "count_toys_%ss" % self.toy_class

    def sname(self):
        return "ct%d" % self.toy_class


class CountDigitOccTask(Task):
    def __init__(self, settings, digit, seq, dbg_learn_parameters):
        input_type = mkListSort(mkRealTensorSort([1, 1, 28, 28]))
        # input_type = mkListSort(mkBoolTensorSort([1, 1]))
        # input_type = mkBoolTensorSort([1, 1])
        output_type = mkRealTensorSort([1, 1])
        fn_sort = mkFuncSort(input_type, output_type)

        super(CountDigitOccTask, self).__init__(fn_sort, settings, seq, dbg_learn_parameters)
        self.digit = digit

    def get_io_examples(self):
        return get_io_examples_count_digit_occ(self.digit, self.settings.train_size, self.settings.val_size)

    def name(self):
        return "count_digit_occ_%ss" % self.digit

    def sname(self):
        return 'c%d' % self.digit