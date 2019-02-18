from typing import List, Dict, Tuple

import numpy as np
from matplotlib import pyplot as plt

from Data.DataProvider import MNISTDataProvider, StreetSignsDataProvider, ToyDataProvider
from Data.DataGenerator import NumpyDataSetIterator
DEFAULT_SEED = 1337  # I know, very unoriginal

def mk_tag(tag: str, content: str, cls: List[str] = [], attribs: Dict = {}):
    cls_str = ' '.join(cls)
    if len(cls_str) > 0:
        cls_str = 'class = "%s"' % cls_str

    attrib_str = ' '.join(['%s="%s"' % (k, v) for k, v in attribs.items()])

    return '<%s %s %s>%s</%s>\n' % (tag, cls_str, attrib_str, content, tag)


def mk_div(content: str, cls: List[str] = []):
    return mk_tag('div', content, cls)


def append_to_file(a_file, content):
    with open(a_file, "a") as fh:
        fh.write(content)


def write_to_file(a_file, content):
    with open(a_file, "w") as fh:
        fh.write(content)


def iterate_diff_training_sizes(train_io_examples, training_data_percentages):
    # assuming all lengths are represented equally
    if issubclass(type(train_io_examples), NumpyDataSetIterator) or \
            type(train_io_examples) == list and issubclass(type(train_io_examples[0]), NumpyDataSetIterator):
        num_of_training_dp = train_io_examples[0].inputs.shape[0]
        # raise NotImplementedError("uhm?!")
        yield train_io_examples, num_of_training_dp
        return
    num_of_training_dp = train_io_examples[0][0].shape[0] if type(train_io_examples) == list else \
        train_io_examples[0].shape[0]

    for percentage in training_data_percentages:
        c_num_items = (percentage * num_of_training_dp) // 100
        if type(train_io_examples) == list:
            c_tr_io_examples = [(t[0][:c_num_items], t[1][:c_num_items]) for t in train_io_examples]
            return_c_num_items = c_num_items * train_io_examples.__len__()
        else:
            c_tr_io_examples = (train_io_examples[0][:c_num_items], train_io_examples[1][:c_num_items])
            return_c_num_items = c_num_items

        yield c_tr_io_examples, return_c_num_items


def save_graph_a2(list_of_tuples, xlabel="Training Dataset Size", ylabel="Accuracy after training",
                  negate_y=False):
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    handles = []  # handles for the plt.legent method
    for tpl in list_of_tuples:
        file_name, legend = tpl
        t_line, = plt.plot(xarray, yarray * (-1 if negate_y else 1), label=legend, marker='o')
        handles.append(t_line)
    plt.legend(handles=handles)
    # plt.show(block=True)
    plt.savefig('Eval/Results/%s.png' % tpl[0])
    plt.close()


def save_graph_a(list_of_tuples, xlabel="Training Dataset Size", ylabel="Accuracy after training",
                 negate_y=False):
    """
    :param list_of_tuples: (filename, name in the plot)
    :return:
    """
    plt.figure()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    handles = []  # handles for the plt.legent method
    for tpl in list_of_tuples:
        file_name, legend = tpl
        nparray = np.load("Eval/Results/%s.npy" % file_name)
        t_line, = plt.plot(nparray[0], nparray[1] * (-1 if negate_y else 1), label=legend, marker='o')
        handles.append(t_line)
    plt.legend(handles=handles)
    # plt.show(block=True)
    plt.savefig('Eval/Results/%s.png' % tpl[0])
    plt.close()


def get_io_examples_recognize_digit(digit, train_size, val_size):
    mnist_data_provider = MNISTDataProvider(random_seed=DEFAULT_SEED)
    mnist_train, mnist_val, mnist_test = mnist_data_provider.split_into_train_and_validation(0, 6)

    train_io_examples = mnist_data_provider.get_batch_counting([digit], 1, train_size, mnist_train,
                                                               single_items=True, return_count_int=False)

    val_io_examples = mnist_data_provider.get_batch_counting([digit], 1, val_size, mnist_val,
                                                             single_items=True, return_count_int=False)

    test_io_examples = mnist_data_provider.get_batch_counting([digit], 1, val_size, mnist_test,
                                                             single_items=True, return_count_int=False)

    return train_io_examples, val_io_examples, test_io_examples


def get_io_examples_classify_digits(train_size, val_size):
    mnist_data_provider = MNISTDataProvider(random_seed=DEFAULT_SEED)
    mnist_train, mnist_val, mnist_test = mnist_data_provider.split_into_train_and_validation(0, 6)

    train_io_examples = mnist_data_provider.get_images_for_classification(mnist_train)
    val_io_examples = mnist_data_provider.get_images_for_classification(mnist_val)
    test_io_examples = mnist_data_provider.get_images_for_classification(mnist_test)
    return train_io_examples, val_io_examples, test_io_examples


def get_io_examples_count_digit_occ(digit, train_size, val_size):
    mnist_data_provider = MNISTDataProvider(random_seed=DEFAULT_SEED)
    mnist_train, mnist_val, mnist_test = mnist_data_provider.split_into_train_and_validation(0, 6)

    train_io_examples = mnist_data_provider.get_batch_count_var_len([digit], train_size, mnist_train,
                                                                    list_lengths=(2, 3, 4, 5),
                                                                    return_count_int=False)

    val_io_examples = mnist_data_provider.get_batch_count_var_len([digit], val_size, mnist_val,
                                                                  list_lengths=(6, 7, 8),
                                                                  return_count_int=False)

    test_io_examples = mnist_data_provider.get_batch_count_var_len([digit], val_size, mnist_test,
                                                                  list_lengths=(6, 7, 8),
                                                                  return_count_int=False)

    return train_io_examples, val_io_examples, test_io_examples


def get_io_examples_sum_digits(train_size, val_size):
    mnist_data_provider = MNISTDataProvider(random_seed=DEFAULT_SEED)
    mnist_train, mnist_val, mnist_test = mnist_data_provider.split_into_train_and_validation(0, 6)

    train_io_examples = mnist_data_provider.get_batch_sum_var_len(train_size, mnist_train, list_lengths=(2, 3, 4, 5),
                                                                  sum_int=False)

    val_io_examples = mnist_data_provider.get_batch_sum_var_len(val_size, mnist_val, list_lengths=(6, 7, 8),
                                                                sum_int=False)

    test_io_examples = mnist_data_provider.get_batch_sum_var_len(val_size, mnist_test, list_lengths=(6, 7, 8),
                                                                sum_int=False)

    return train_io_examples, val_io_examples, test_io_examples


def get_io_examples_recognize_toy(toy, train_size, val_size):
    # toy_data_generator = ToyDataProvider(random_seed=DEFAULT_SEED)
    global _toy_data_provider
    if _toy_data_provider is None:
        toy_data_generator = ToyDataProvider(random_seed=DEFAULT_SEED)  # TODO: Do not recreate ToyDataProvider
        _toy_data_provider = toy_data_generator
    else:
        toy_data_generator = _toy_data_provider
    toy_train_dictionary, toy_val_dictionary, toy_test_dictionary = toy_data_generator.split_into_train_and_validation(0, 10)

    train_io_examples = toy_data_generator.get_batch_counting([toy], 1, train_size, toy_train_dictionary,
                                                              single_items=True, return_count_int=False)
    val_io_examples = toy_data_generator.get_batch_counting([toy], 1, val_size, toy_val_dictionary,
                                                            single_items=True, return_count_int=False)

    test_io_examples = toy_data_generator.get_batch_counting([toy], 1, val_size, toy_test_dictionary,
                                                            single_items=True, return_count_int=False)

    return train_io_examples, val_io_examples, test_io_examples

_toy_data_provider = None

def get_io_examples_count_toys(toy_class, train_size, val_size):
    global _toy_data_provider
    if _toy_data_provider is None:
        toy_data_generator = ToyDataProvider(random_seed=DEFAULT_SEED)  # TODO: Do not recreate ToyDataProvider
        _toy_data_provider = toy_data_generator
    else:
        toy_data_generator = _toy_data_provider

    toy_train_dictionary, toy_val_dictionary, toy_test_dictionary = toy_data_generator.split_into_train_and_validation(0, 10)
    train_io_examples = toy_data_generator.get_batch_count_var_len([toy_class], train_size, toy_train_dictionary,
                                                                   list_lengths=(2, 3, 4, 5),
                                                                   return_count_int=False)

    val_io_examples = toy_data_generator.get_batch_count_var_len([toy_class], val_size, toy_val_dictionary,
                                                                 list_lengths=((6, 7, 8)),
                                                                 return_count_int=False)

    test_io_examples = toy_data_generator.get_batch_count_var_len([toy_class], val_size, toy_test_dictionary,
                                                                 list_lengths=((6, 7, 8)),
                                                                 return_count_int=False)

    return train_io_examples, val_io_examples, test_io_examples


"""
def get_io_examples_classify_speed(train_size, val_size):
    data_provider = StreetSignsDataProvider()
    train, val, test = data_provider.split_into_train_and_validation(0, 6)

    train_io_examples = data_provider.get_images_for_classification(train)
    val_io_examples = data_provider.get_images_for_classification(val)
    test_io_examples = data_provider.get_images_for_classification(test)
    return train_io_examples, val_io_examples, test_io_examples


def get_io_examples_shortest_path_speed_maze(train_size, val_size):
    # val_to_class_dict's keys are the values, which will populate the maze. (3, 5, 7)
    # val_to_class_dict's vals are the image classes for every value in the maze
    # img_class_start, img_class_goal are the values for the, surprise, start and goal
    data_provider = StreetSignsDataProvider()
    img_dict_train, img_dict_val, img_dict_test = data_provider.split_into_train_and_validation(0, 6)

    train_io_examples = data_provider.get_batch_maze(num_samples=train_size, grid_size=3,
                                                     val_to_class_dict={3: 1, 5: 2, 7: 4},
                                                     img_class_start=26, img_class_goal=14,
                                                     img_dictionary=img_dict_train)
    val_io_examples = data_provider.get_batch_maze(num_samples=val_size, grid_size=3,
                                                   val_to_class_dict={3: 1, 5: 2, 7: 4},
                                                   img_class_start=26, img_class_goal=14,
                                                   img_dictionary=img_dict_val)

    test_io_examples = data_provider.get_batch_maze(num_samples=val_size, grid_size=3,
                                                   val_to_class_dict={3: 1, 5: 2, 7: 4},
                                                   img_class_start=26, img_class_goal=14,
                                                   img_dictionary=img_dict_test)

    return train_io_examples, val_io_examples
"""

def get_io_examples_shortest_path_mnist_maze(batch_size,
                                             num_3x3_mazes = 70000, num_4x4_mazes = 1000000, num_5x5_mazes = 10000):

    mnist_dict_val_to_class = {9: 9, 7: 7, 5: 5, 3: 3, 1: 1, 0: 1}

    mnist_data_provider = MNISTDataProvider(expand=True, random_seed=DEFAULT_SEED)
    mnist_img_dict_train, mnist_img_dict_val, mnist_img_dict_test = mnist_data_provider.split_into_train_and_validation(0, 6,
                                                                                                   shuffleFirst=True)
    if num_3x3_mazes > 1:
        maze_io_train3x3, _, _ = mnist_data_provider.get_tr_val_test_generators_maze(num_3x3_mazes, 1,
                                                                                     batch_size, grid_size=3,
                                                                                     val_to_class_dict=mnist_dict_val_to_class,
                                                                                     img_dictionary_tr=mnist_img_dict_train,
                                                                                     img_dictionary_val=mnist_img_dict_val,
                                                                                     img_dictionary_test=mnist_img_dict_test,
                                                                                     img_class_start=0,
                                                                                     img_class_goal=1,
                                                                                     only_last_shortest_path=False)
    maze_io_train4x4, _, _ = mnist_data_provider.get_tr_val_test_generators_maze(num_4x4_mazes, 1,
                                                                                 batch_size, grid_size=4,
                                                                                 val_to_class_dict=mnist_dict_val_to_class,
                                                                                 img_dictionary_tr=mnist_img_dict_train,
                                                                                 img_dictionary_val=mnist_img_dict_val,
                                                                                 img_dictionary_test=mnist_img_dict_test,
                                                                                 img_class_start=0,
                                                                                 img_class_goal=1,
                                                                                 only_last_shortest_path=False)
    _, maze_io_val5x5, maze_io_test5x5 = mnist_data_provider.get_tr_val_test_generators_maze(1, num_5x5_mazes,
                                                                                             batch_size, grid_size=5,
                                                                                             val_to_class_dict=mnist_dict_val_to_class,
                                                                                             img_dictionary_tr=mnist_img_dict_train,
                                                                                             img_dictionary_val=mnist_img_dict_val,
                                                                                             img_dictionary_test=mnist_img_dict_test,
                                                                                             img_class_start=0,
                                                                                             img_class_goal=1,
                                                                                             only_last_shortest_path=False)
    training_iteratos = [maze_io_train3x3, maze_io_train4x4] if num_3x3_mazes > 1 else [ maze_io_train4x4]
    validation_iterator = maze_io_val5x5
    test_iterator = maze_io_test5x5

    return training_iteratos, validation_iterator, test_iterator


def get_io_examples_regress_speed_mnist():
    iv = 65.  # iv = initial value
    mnist_dict_class_to_val = {9: (9, iv), 7: (7, iv), 5: (5, iv), 3: (3, iv), 1: (1, iv), 0: (1, 1)}  # 0 for start, 1 for end

    mnist_data_provider = MNISTDataProvider(expand=True, random_seed=DEFAULT_SEED)
    mnist_train, mnist_val, mnist_test = mnist_data_provider.split_into_train_and_validation(0, 6)

    train_io_examples = mnist_data_provider.get_images_for_regression(mnist_train, mnist_dict_class_to_val)
    val_io_examples = mnist_data_provider.get_images_for_regression(mnist_val, mnist_dict_class_to_val)
    test_io_examples = mnist_data_provider.get_images_for_regression(mnist_test, mnist_dict_class_to_val)

    return train_io_examples, val_io_examples, test_io_examples


def get_io_examples_regress_speed_street_sign():
    iv = 65.  # iv = initial value
    dict_class_to_val = {1: (9, iv), 2: (7, iv), 4: (5, iv), 5: (3, iv), 7: (1, iv), 26: (1, iv), 14: (1, iv)}  # 26 for start, 14 for end
    #

    data_provider = StreetSignsDataProvider(random_seed=DEFAULT_SEED)
    img_dict_train, img_dict_val, img_dict_test = data_provider.split_into_train_and_validation(0, 6)

    train_io_examples = data_provider.get_images_for_regression(img_dict_train, dict_class_to_val)
    val_io_examples = data_provider.get_images_for_regression(img_dict_val, dict_class_to_val)
    test_io_examples = data_provider.get_images_for_regression(img_dict_test, dict_class_to_val)

    return train_io_examples, val_io_examples, test_io_examples


def get_io_examples_shortest_path_street_sign_maze(batch_size,
                                                   num_3x3_mazes=70000, num_4x4_mazes=1000000, num_5x5_mazes=10000):

    dict_val_to_class = {9: 1, 7: 2, 5: 4, 3: 5, 1: 7}

    data_provider = StreetSignsDataProvider(random_seed=DEFAULT_SEED)
    img_dict_train, img_dict_val, img_dict_test = data_provider.split_into_train_and_validation(0, 10)
    if num_3x3_mazes > 1:
        maze_io_train3x3, _, _ = data_provider.get_tr_val_test_generators_maze(num_3x3_mazes, 1,
                                                                               batch_size, grid_size=3,
                                                                               val_to_class_dict=dict_val_to_class,
                                                                               img_dictionary_tr=img_dict_train,
                                                                               img_dictionary_val=img_dict_val,
                                                                               img_dictionary_test=img_dict_test,
                                                                               img_class_start=26,
                                                                               img_class_goal=14,
                                                                               only_last_shortest_path=False)
    maze_io_train4x4, _, _ = data_provider.get_tr_val_test_generators_maze(num_4x4_mazes, 1,
                                                                           batch_size, grid_size=4,
                                                                           val_to_class_dict=dict_val_to_class,
                                                                           img_dictionary_tr=img_dict_train,
                                                                           img_dictionary_val=img_dict_val,
                                                                           img_dictionary_test=img_dict_test,
                                                                           img_class_start=26,
                                                                           img_class_goal=14,
                                                                           only_last_shortest_path=False)
    _, maze_io_val5x5, maze_io_test_5x5 = data_provider.get_tr_val_test_generators_maze(1, num_5x5_mazes,
                                                                                        batch_size, grid_size=5,
                                                                                        val_to_class_dict=dict_val_to_class,
                                                                                        img_dictionary_tr=img_dict_train,
                                                                                        img_dictionary_val=img_dict_val,
                                                                                        img_dictionary_test=img_dict_test,
                                                                                        img_class_start=26,
                                                                                        img_class_goal=14,
                                                                                        only_last_shortest_path=False)
    # training_iteratos = [maze_io_train3x3, maze_io_train4x4]
    training_iteratos = [maze_io_train3x3, maze_io_train4x4] if num_3x3_mazes > 1 else [maze_io_train4x4]
    validation_iterator = maze_io_val5x5
    test_iterator = maze_io_test_5x5

    return training_iteratos, validation_iterator, test_iterator