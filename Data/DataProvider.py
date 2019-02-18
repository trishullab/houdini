import numpy as np
from random import randint
import random
import matplotlib.pyplot as plt
import torchvision
from Data.MazeGenerator import MazeGenerator
from Data.DataGenerator import NumpyDataSetIterator, ImageGraphDataGenerator

import re
from PIL import Image
import glob
import pickle
import abc
import urllib.request
import zipfile, gzip

from datetime import datetime, time

# from scipy.misc import toimage
# from skimage.color import rgb2yuv, yuv2rgb
from skimage.exposure import rescale_intensity
from skimage.transform import resize
from Data.smallnorb.dataset import SmallNORBDataset
import torch.nn as nn
import os

# np.random.seed(1337)


class DataProvider(object):
    def __init__(self, get_test=True, random_seed=None):
        """
        :param get_test: If true, loads the test dataset as well as the training one
        :param random_seed: a number, used as a random seed for split. Then random_seed+1 is used for data generation
        """
        self.ds_images = {
            "sorted_images": None,
            "sorted_labels": None,
            "labels_count": None,
            "labels_indices": None,
            "num_labels": None
        }
        self.ds_images_test = {
            "sorted_images": None,
            "sorted_labels": None,
            "labels_count": None,
            "labels_indices": None,
            "num_labels": None
        }
        self.image_size = self.get_image_size()
        self.get_test = get_test
        self.random_seed = random_seed

    def get_num_classes(self):
        return self.ds_images["num_labels"]

    @abc.abstractmethod
    def load_data(self):
        """
        image_size = self.get_image_size()
        train_all_images = np array, type=np.float32, shape=(-1, image_ch, image_size, image_size)
        train_all_labels = np array, type=np.float32, shape=(-1, 1)
        train_all_labels is assumed to be classes {0., 1., 2. etc...}
        :return: returns train_all_labels, train_all_images
        :return: train_all_labels, train_all_images, test_all_labels, test_all_images (if training)
        """

    @abc.abstractmethod
    def get_image_size(self):
        """
        :return: returns the image size along 1 dimension (assuming square images)
        """

    @abc.abstractmethod
    def get_image_channels_num(self):
        """
        :return: returns the number of channels of the image
        """


    @staticmethod
    def shuffle_in_unison(a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        if c is not None:
            np.random.set_state(rng_state)
            np.random.shuffle(c)

    @staticmethod
    def split_matrix_into_train_and_validation(mat, val_start_indx, val_len):
        t_1 = mat[0:val_start_indx]
        v = mat[val_start_indx:val_start_indx + val_len]
        t_2 = mat[val_start_indx + val_len:]
        t = np.vstack((t_1, t_2))
        return t, v

    def prepare_image_data(self):
        """
        Prepares the images, used for subsequent tasks.
        :return:
        """
        if self.get_test:
            train_all_images, train_all_labels, test_all_images, test_all_labels = self.load_data()
        else:
            train_all_images, train_all_labels = self.load_data()

        self.mean = train_all_images.mean()
        self.std = train_all_images.std()
        train_all_images = (train_all_images - self.mean) / self.std
        if self.get_test:
            test_all_images = (test_all_images - self.mean) / self.std

        # print(train_all_images.min())
        # print(train_all_images.max())

        def sort_data(dictionary, np_images, np_labels):
            # split into 10 data-label pairs, 1 pair for each digit
            labels_sorted_order = np.argsort(np_labels, axis=0)

            dictionary['sorted_labels'] = np_labels[labels_sorted_order]
            dictionary['sorted_images'] = np_images[labels_sorted_order]

            items_unique, items_counts = np.unique(np_labels, return_counts=True)
            dictionary['num_labels'] = items_unique.size
            dictionary['labels_count'] = dict(zip(items_unique, items_counts))
            dictionary['labels_indices'] = {0: {'first': 0, 'last': items_counts[0] - 1}}

            for i in range(1, dictionary['num_labels']):
                first_indx = dictionary['labels_indices'][i - 1]['last'] + 1
                dictionary['labels_indices'][i] = {'first': first_indx, 'last': first_indx + items_counts[i] - 1}

        sort_data(self.ds_images, train_all_images, train_all_labels)
        if self.get_test:
            sort_data(self.ds_images_test, test_all_images, test_all_labels)

    def split_into_train_and_validation(self, k, total_parts_count, shuffleFirst=False):
        if self.ds_images['sorted_images'] is None:
            print("calling prepare_data()")
            self.prepare_image_data()

        if self.get_test:
            train = {}
            validation = {}
            test = {}
            # for each class
            for class_id in range(self.ds_images['num_labels']):
                first_index = self.ds_images['labels_indices'][class_id]['first']
                last_index = self.ds_images['labels_indices'][class_id]['last']
                relevant_data = self.ds_images['sorted_images'][first_index:last_index + 1]
                number_of_datapoints = relevant_data.shape[0]
                # train[class_id] = relevant_data

                num_of_validation_datapoints = number_of_datapoints // total_parts_count
                val_start_indx = k * num_of_validation_datapoints
                t_dp, v_dp = DataProvider.split_matrix_into_train_and_validation(relevant_data, val_start_indx,
                                                                                 num_of_validation_datapoints)
                train[class_id] = t_dp
                validation[class_id] = v_dp

                first_index_test = self.ds_images_test['labels_indices'][class_id]['first']
                last_index_test = self.ds_images_test['labels_indices'][class_id]['last']
                relevant_data_test = self.ds_images_test['sorted_images'][first_index_test:last_index_test + 1]
                test[class_id] = relevant_data_test

            return train, validation, test

        # the case where self.get_test=False, isn't implemented atm
        raise NotImplementedError

    @staticmethod
    def get_random_image(class_id, image_dictionary):
        first_index = 0
        last_index = image_dictionary[class_id].shape[0] - 1

        indx = randint(first_index, last_index)
        img = image_dictionary[class_id][indx][0]
        return img

    @staticmethod
    def shuffle_lists_together(a, b):
        c = list(zip(a, b))
        random.shuffle(c)
        a, b = zip(*c)
        return a, b

    @staticmethod
    def to_one_of_k(int_targets, num_classes):
        b = np.zeros((int_targets.shape[0], num_classes))
        int_targets = np.reshape(a=int_targets, newshape=(b.shape[0],))
        b[np.arange(b.shape[0]), int_targets] = 1
        return b

    def get_images_for_classification(self, image_dictionary, classes=None, remap_classes=True):
        # remap_classes = False
        if classes is None:
            remap_classes = False
            classes = range(self.ds_images["num_labels"])
        else:
            if remap_classes:
                classes_dict = {}
                for c_index, c_class in enumerate(classes):
                    classes_dict[c_class] = c_index

        labels = np.zeros((0,), np.int)
        img_dim = self.get_image_size()
        img_ch = self.get_image_channels_num()

        dp = np.zeros((0, 1, img_ch, img_dim, img_dim), np.float32)

        #print("get_images_for_classification")
        #print(dp.shape)
        for image_label_int, val in image_dictionary.items():
            # print(val.shape)
            if image_label_int in classes:
                dp = np.vstack((dp, val))
                label_to_use = image_label_int if not remap_classes else classes_dict[image_label_int]
                labels = np.hstack((labels, np.ones((val.shape[0],), np.int) * label_to_use))

        labels = labels.reshape((-1, 1))
        self.shuffle_in_unison(dp, labels, c=None)
        labels = labels.reshape((-1,))

        dp = np.reshape(dp, (-1, img_ch, img_dim, img_dim))
        return dp, labels

    def get_images_for_regression(self, image_dictionary, class_value_dict):
        """
        Returns regression labels. instead of labels = ([batch_size,], np.int], it's (batch_size, 1), np.float32)
        :param image_dictionary:
        :param class_value_dict: {class_idx : value}
        :return:
        """
        dp, labels = self.get_images_for_classification(image_dictionary, class_value_dict.keys(), remap_classes=False)
        labels = labels.astype(np.float32).reshape(-1, 1)

        sample_class_value = list(class_value_dict.values())[0]

        if type(sample_class_value) is not tuple:
            for key, value in class_value_dict.items():
                labels[labels == float(key)] = float(value)

            return dp, labels
        else:
            num_dp = labels.shape[0]
            new_labels = np.zeros((num_dp, 2), np.float32)

            for l_idx in range(num_dp):
                val = labels[l_idx]
                new_labels[l_idx] = class_value_dict[val[0]]

            return dp, new_labels

    def show_image(self, x):
        """
        Stitches a list of images horizontally and presents them.
        :param x: a numpy array of size [list_length, color_ch, img_w, img_h]
        """
        if x.shape.__len__() == 5: # if it's a grid of images
            img_dim = self.get_image_size()
            img_ch = self.get_image_channels_num()
            x = np.transpose(x, (0, 1, 3, 4, 2))

            grid_h = x.shape[0]
            grid_w = x.shape[1]
            # go through each row and concatenate horizontaly
            image = np.zeros((0, grid_w*img_dim, img_ch), np.float32)
            for h in range(grid_h):
                h_image = np.zeros((img_dim, 0, img_ch), np.float32)
                for w in range(grid_w):
                    h_image = np.hstack((h_image, x[h, w]))
                image = np.vstack((image, h_image))
            img = image
        elif x.shape.__len__() == 4: # if it's a list of images
            # convert all images from CHW to HWC
            x = x.swapaxes(1, 2).swapaxes(2, 3)
            num_images = x.shape[0]
            list_images = np.split(x, num_images, axis=0)
            list_images = [a.squeeze(axis=0) for a in list_images]

            img = np.hstack(list_images)
            """
            if self.get_image_channels_num() == 1:
                img = img.squeeze(axis=2)
            else:
                img = img * self.std + self.mean
                img = img * 0.5 + 0.5
            """
        else:
            img = x.swapaxes(0, 1).swapaxes(1, 2)

        if self.get_image_channels_num() == 1:
            img = img.squeeze(axis=2)
        else:
            img = img * self.std + self.mean
            img = img * 0.5 + 0.5

        print(img.shape)
        imgplot = plt.imshow(img, cmap=plt.get_cmap("gray"))
        plt.show(block=True)

    def get_batch_counting(self, classes_to_count, list_length, batch_size, digit_dictionary, classes_to_use=None,
                           single_items=False, return_count_int=True, random_states=None, return_starting_rand_states=False):
        """
        :param classes_to_count: a list of classes to count
        :param list_length: what's the list length
        :param batch_size:
        :param digit_dictionary: the dictionary of images {class -> images}
        :param classes_to_use: the classes that can be put in the list. if None, defaults to all the classes
        :param single_items: set to True, if list_length is equal to 1. this is used for a 0/1 classifier of classes_to_count
        :param return_count_int: whether to return the labels as int or float
        :return:
        """
        starting_rndm_states = {"numpy": np.random.get_state(), "python": random.getstate()}
        if random_states is not None:
            # print("SETTING RANDOM STATES")
            np.random.set_state(random_states["numpy"])
            random.setstate(random_states["python"])
        elif self.random_seed is not None:
            random.seed(self.random_seed+1+classes_to_count[0]+batch_size)
            np.random.seed(self.random_seed+1+classes_to_count[0]+batch_size)

        pvals = np.ones((list_length + 1,), np.float32) / (list_length + 1)
        proportions = np.random.multinomial(batch_size, pvals)

        # available_cell_indices = list(range(total_images_count))
        if classes_to_use is None:
            classes_to_use = range(self.get_num_classes())
        classes_to_use_but_not_count = [i for i in classes_to_use if i not in classes_to_count]

        img_dim = self.get_image_size()
        img_ch_num = self.get_image_channels_num()

        batch_datapoints = np.zeros((batch_size, list_length, img_ch_num, img_dim, img_dim), dtype=np.float32)
        batch_count_labels = np.zeros((batch_size, 1), dtype=np.int32)

        current_samples_index = -1
        # for each group/label 0 - count_up_to, e.g. 0-5
        for c_count_label in range(list_length + 1):
            # create samples_per_number samples
            for sample_index in range(proportions[c_count_label]):

                current_samples_index += 1
                images = []

                # c_count_label is the current count.
                # first, add c_count_label number of images from classes_to_count
                for i in range(c_count_label):
                    c_img_class = random.choice(classes_to_count)
                    c_img = self.get_random_image(c_img_class, digit_dictionary)

                    # the index is random since available_cell_indices was shuffled
                    images.append(c_img)

                # fill the rest of the list with images of random classes
                num_images_left_to_add = list_length - c_count_label
                for i in range(num_images_left_to_add):
                    c_img_class = random.choice(classes_to_use_but_not_count)
                    c_img = self.get_random_image(c_img_class, digit_dictionary)
                    images.append(c_img)

                # shuffle the array of images and then convert to a numpy array
                random.shuffle(images)

                # add the images to the dataset
                batch_datapoints[current_samples_index] = images
                batch_count_labels[current_samples_index] = c_count_label

        self.shuffle_in_unison(batch_datapoints, batch_count_labels, None)

        batch_count_labels = np.reshape(batch_count_labels, [-1, 1])
        batch_count_labels_int = batch_count_labels.astype(np.int)
        batch_count_labels_float = batch_count_labels.astype(np.float32)
        # batch_count_labels_one_hot = to_one_of_k(batch_count_labels, count_up_to + 1)

        if list_length == 1 and single_items:
            batch_datapoints = batch_datapoints.reshape((batch_size, img_ch_num, img_dim, img_dim))

        return_labels = (batch_count_labels_float if not return_count_int else batch_count_labels_int.reshape(-1))

        # print("Random Number: {}".format(random.randrange(0, 101, 2)))
        # print(np.random.rand(1, 5))
        # random.seed(datetime.now())

        if return_starting_rand_states:
            return (batch_datapoints, return_labels), starting_rndm_states
        return batch_datapoints, return_labels

    def get_batch_count_var_len(self, classes_to_count, batch_size, digit_dictionary, list_lengths=(1, 2, 3),
                                classes_to_use=None, return_count_int=True,
                                random_states=None, return_starting_rand_states=False):
        starting_rndm_states = {"numpy": np.random.get_state(), "python": random.getstate()}

        if random_states is not None:
            # print("SETTING RANDOM STATES")
            np.random.set_state(random_states["numpy"])
            random.setstate(random_states["python"])
        elif self.random_seed is not None:
            random.seed(self.random_seed+1+classes_to_count[0]+batch_size)
            np.random.seed(self.random_seed+1+classes_to_count[0]+batch_size)

        bs_per_len = batch_size // list_lengths.__len__()

        result = []
        for ll in list_lengths:
            c_rndm_states = {"numpy": np.random.get_state(), "python": random.getstate()}
            result.append(self.get_batch_counting(classes_to_count, ll, bs_per_len, digit_dictionary, classes_to_use,
                                                  single_items=False, return_count_int=return_count_int,
                                                  random_states=c_rndm_states))
        # random.seed(datetime.now())

        if return_starting_rand_states:
            return result, starting_rndm_states
        return result

    def get_batch_sum(self, batch_size, img_dictionary, list_length=5, sum_int=False):
        """
        :param batch_size:
        :param img_dictionary:
        :param list_length:
        :param sum_int: whether to return the labels as a numpy array of type int or float
        :return:
        """
        classes_available = range(self.get_num_classes())
        img_dim = self.get_image_size()
        img_ch_num = self.get_image_channels_num()

        batch_datapoints = np.zeros((batch_size, list_length, img_ch_num, img_dim, img_dim), dtype=np.float32)
        batch_sum_labels = np.zeros((batch_size, 1), dtype=np.int32)

        for sample_index in range(batch_size):
            images = []
            c_sum = 0

            for list_index in range(list_length):
                random_class = random.choice(classes_available)

                img = self.get_random_image(random_class, img_dictionary)
                images.append(img)
                c_sum += random_class

            batch_datapoints[sample_index] = images
            batch_sum_labels[sample_index] = c_sum

        batch_sum_labels_float = batch_sum_labels.astype(np.float32)

        return_sum_labels = batch_sum_labels_float if not sum_int else batch_sum_labels.reshape(-1)
        return batch_datapoints, return_sum_labels

    def get_batch_sum_var_len(self, batch_size, img_dictionary, list_lengths=(1, 2, 3), sum_int=True,
                              random_states=None, return_starting_rand_states=False):
        starting_rndm_states = {"numpy": np.random.get_state(), "python": random.getstate()}
        if random_states is not None:
            # print("SETTING RANDOM STATES")
            np.random.set_state(random_states["numpy"])
            random.setstate(random_states["python"])

        bs_per_len = batch_size // list_lengths.__len__()

        result = [self.get_batch_sum(bs_per_len, img_dictionary, ll, sum_int)
                  for ll in list_lengths]

        # random.seed(datetime.now())
        if return_starting_rand_states:
            return result, starting_rndm_states
        return result

    def get_batch_maze(self, num_samples, grid_size, val_to_class_dict, img_dictionary,
                       img_class_start, img_class_goal):
        """
        :param num_samples:
        :param grid_size:
        :param val_to_class_dict: {maze_value -> image_class}
        :param img_dictionary:
        :return:
        """
        # convert the keys to floats
        val_to_class_dict = dict(zip([float(k) for k in val_to_class_dict.keys()], list(val_to_class_dict.values())))
        maze, maze_labels, start, goal = MazeGenerator.generate_data(num_samples, grid_size,
                                                                     list(val_to_class_dict.keys()))
        # print(maze)

        img_dim = self.get_image_size()
        img_ch = self.get_image_channels_num()
        maze_w_imgs = np.zeros(shape=list(maze.shape)+[img_ch, img_dim, img_dim], dtype=np.float32)

        for b in range(maze.shape[0]):
            for r in range(maze.shape[1]):
                for c in range(maze.shape[2]):
                    if r==start[0] and c==start[1]:
                        c_img_class = img_class_start
                    elif r==goal[0] and c==goal[1]:
                        c_img_class = img_class_goal
                    else:
                        c_maze_value = maze[b, r, c]
                        c_img_class = val_to_class_dict[c_maze_value]
                    c_rndm_img = self.get_random_image(c_img_class, img_dictionary)
                    maze_w_imgs[b, r, c] = c_rndm_img

        print(maze_w_imgs.shape)

        """
        max_element_value = max(elements)
        v = np.ones_like(maze, dtype=np.float32) * grid_size*grid_size*max_element_value # TODO: chance this initial value to something more permanent
        new_maze = np.stack((maze, v),axis=1)
        """
        return maze_w_imgs, maze_labels

    def get_tr_val_test_generators_maze(self, num_samples_tr, num_samples_val, batch_size, grid_size, val_to_class_dict,
                                        img_dictionary_tr, img_dictionary_val, img_dictionary_test, img_class_start, img_class_goal,
                                        only_last_shortest_path=True):
        """
        :param num_samples:
        :param grid_size:
        :param val_to_class_dict: {maze_value -> image_class}
        :param img_dictionary:
        :return:
        """
        # convert the keys to floats
        val_to_class_dict = dict(zip([float(k) for k in val_to_class_dict.keys()], list(val_to_class_dict.values())))
        num_unique_items = val_to_class_dict.items().__len__()**(grid_size*grid_size-2)
        if num_unique_items < 500000:
            data = MazeGenerator.generate_data_tr_val_test(num_samples_tr, num_samples_val, grid_size,
                                                           list(val_to_class_dict.keys()),
                                                           only_last_shortest_path=only_last_shortest_path)
        else:
            data = MazeGenerator.generate_data_tr_val_test_faster(num_samples_tr, num_samples_val, grid_size,
                                                                  list(val_to_class_dict.keys()),
                                                                  only_last_shortest_path=only_last_shortest_path)
        tr_maze, tr_maze_labels, val_maze, val_maze_labels, test_maze, test_maze_labels = data
        # maze, maze_labels, start, goal = MazeGenerator.generate_data(num_samples, grid_size,
        #                                                              list(val_to_class_dict.keys()))
        # print(maze)
        generator_tr = ImageGraphDataGenerator(tr_maze, tr_maze_labels, batch_size, grid_size,
                                               self, img_dictionary_tr, val_to_class_dict, img_class_start, img_class_goal)
        generator_val = ImageGraphDataGenerator(val_maze, val_maze_labels, batch_size, grid_size,
                                                self, img_dictionary_val, val_to_class_dict, img_class_start,
                                                img_class_goal)
        generator_test = ImageGraphDataGenerator(test_maze, test_maze_labels, batch_size, grid_size,
                                                self, img_dictionary_test, val_to_class_dict, img_class_start,
                                                img_class_goal)

        return generator_tr, generator_val, generator_test

    @staticmethod
    def download_if_needed(dataset_name, directory, filepath_to_check, urls):
        """
        :param directory: to store the files in
        :param filepath_to_check:
        :param zips: list of urls
        :return:
        """

        if not os.path.exists(directory):
            os.makedirs(directory)

        if not os.path.exists(filepath_to_check):
            print("Dataset {} doesn't exist. Will attempt to download it next.".format(dataset_name))
            for c_download_url in urls:
                c_archive_filename = os.path.basename(c_download_url)
                c_archive_filepath = "{}/{}".format(directory, c_archive_filename)

                if not os.path.exists(c_archive_filepath):
                    print("Downloading {}".format(c_archive_filename))
                    urllib.request.urlretrieve(c_download_url, c_archive_filepath)

                if c_archive_filename[-2:] == "gz":
                    new_filename = "{}/{}".format(directory, c_archive_filename[:-3])
                    with gzip.GzipFile(c_archive_filepath, 'rb') as f_in:
                        with open(new_filename, 'wb') as f_out:
                            f_out.write(f_in.read())
                else:
                    zip_ref = zipfile.ZipFile(c_archive_filepath, 'r')
                    zip_ref.extractall(directory)
                    zip_ref.close()



class MNISTDataProvider(DataProvider):
    MNIST_DIM = 28
    MNIST_CH_NUM = 1

    def __init__(self, get_test: bool=True, expand=False, random_seed=None):
        if random_seed is not None:
            random_seed += 1 # add a data-set specific value to the random seed
        super().__init__(get_test=get_test, random_seed=random_seed)
        self.expand = expand
        if self.expand:
            self.MNIST_CH_NUM = 3
            self.MNIST_DIM = 28

    def load_data(self):
        """
        image_size = self.get_image_size()
        train_all_images = np array, type=np.float32, shape=(-1, 1, image_size, image_size)
        train_all_labels = np array, type=np.float32, shape=(-1, 1)
        train_all_labels is assumed to be classes {0., 1., 2. etc...}
        :return: returns train_all_labels, train_all_images
        """
        if not os.path.exists("Data"):
            raise FileNotFoundError

        def resize_mnist(images):
            images = images / 255.
            expanded_images = np.ones((images.shape[0], self.MNIST_CH_NUM, self.MNIST_DIM, self.MNIST_DIM), np.float32)
            for i in range(images.shape[0]):
                img = images[i][0]
                # print(img.max())
                # print(img.min())

                expanded_images[i][0] = img  # resize(img, output_shape=(self.MNIST_DIM, self.MNIST_DIM))
            return expanded_images

        mnist_trainset = torchvision.datasets.MNIST("Data/MNIST", download=True)
        train_all_labels = np.expand_dims(mnist_trainset.train_labels.numpy(), axis=1)
        train_all_images = mnist_trainset.train_data.numpy().reshape((-1, 1, 28, 28)).astype(np.float32)
        if self.expand:
            train_all_images = resize_mnist(train_all_images)

        if self.get_test:
            mnist_testset = torchvision.datasets.MNIST("Data/MNIST", train=False, download=True)
            test_all_labels = np.expand_dims(mnist_testset.test_labels.numpy(), axis=1)
            test_all_images = mnist_testset.test_data.numpy().reshape((-1, 1, 28, 28)).astype(np.float32)
            if self.expand:
                test_all_images = resize_mnist(test_all_images)
            return train_all_images, train_all_labels, test_all_images, test_all_labels

        return train_all_images, train_all_labels

    def get_image_size(self):
        """
        :return: returns the image size along 1 dimension (assuming square images)
        """
        return self.MNIST_DIM

    def get_image_channels_num(self):
        return self.MNIST_CH_NUM


class ToyDataProvider(DataProvider):
    TOY_DIM = 28  # 96
    TOY_CH_NUM = 1

    def __init__(self, get_test: bool=True, random_seed=None):
        if random_seed is not None:
            random_seed += 2 # add a data-set specific value to the random seed
        super().__init__(get_test=get_test, random_seed=random_seed)

    def load_data(self):
        """
        image_size = self.get_image_size()
        train_all_images = np array, type=np.float32, shape=(-1, 1, image_size, image_size)
        train_all_labels = np array, type=np.float32, shape=(-1, 1)
        train_all_labels is assumed to be classes {0., 1., 2. etc...}
        :return: returns train_all_labels, train_all_images
        """
        if not os.path.exists("Data"):
            raise FileNotFoundError
        dataset_name = "smallnorb"
        directory = "Data/smallnorb"
        training_file = "{}/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat".format(directory)
        urls = [
            "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz",
            "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz",
            "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz",
            "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz",
            "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz",
            "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz"
        ]
        # zips = [("zip{}.gz".format(idx), url) for idx, url in enumerate(urls)]
        super(ToyDataProvider, self).download_if_needed(dataset_name, directory, training_file, urls)

        def pre_process_toy_data(toy_data):
            all_images = np.array([image.image_lt for image in toy_data], np.float32)
            all_labels = np.array([image.category for image in toy_data], np.float32)

            all_images = all_images / 255.

            all_images_smaller = np.zeros((all_images.shape[0], 28, 28), np.float32)
            for i in range(all_images.shape[0]):
                c_img = all_images[i]
                resized_img = resize(c_img, output_shape=(28, 28))
                all_images_smaller[i] = resized_img

            all_images = all_images_smaller
            all_images = np.expand_dims(all_images, 1)
            all_labels = np.expand_dims(all_labels, 1)
            return all_images, all_labels

        smallNorbDataset = SmallNORBDataset(dataset_root='./Data/smallnorb/', just_train=(not self.get_test))
        training_data = smallNorbDataset.data['train']
        train_all_images, train_all_labels = pre_process_toy_data(training_data)
        if self.get_test:
            testing_data = smallNorbDataset.data['test']
            test_all_images, test_all_labels = pre_process_toy_data(testing_data)
            return train_all_images, train_all_labels, test_all_images, test_all_labels

        return train_all_images, train_all_labels

        # return np.zeros((24300, 1, 96, 96), np.float32), np.zeros((24300, 1), np.float32)

    def get_image_size(self):
        """
        :return: returns the image size along 1 dimension (assuming square images)
        """
        return self.TOY_DIM

    def get_image_channels_num(self):
        return self.TOY_CH_NUM


class StreetSignsDataProvider(DataProvider):

    SS_DIM = 28
    SS_CH_NUM = 3

    def __init__(self, get_test: bool=True, random_seed=None):
        if random_seed is not None:
            random_seed += 3  # add a data-set specific value to the random seed
        super().__init__(get_test=get_test, random_seed=random_seed)

    def preprocess_data(self, X):
        # Make all image array values fall within the range -1 to 1
        # Note all values in original images are between 0 and 255, as uint8
        X = X.astype('float32')
        X = (X - 128.) / 128.
        # print((X).mean())
        # print((X).std())
        return X

    def load_data(self):
        """
        image_size = self.get_image_size()
        train_all_images = np array, type=np.float32, shape=(-1, 1, image_size, image_size)
        train_all_labels = np array, type=np.float32, shape=(-1, 1)
        train_all_labels is assumed to be classes {0., 1., 2. etc...}
        :return: returns train_all_labels, train_all_images
        """
        if not os.path.exists("Data"):
            raise FileNotFoundError
        dataset_name = "GTSRB"
        directory = "Data/GTSRB"
        training_file = '{}/lab 2 data/train.p'.format(directory)
        test_file = '{}/lab 2 data/test.p'.format(directory)
        zips = ["https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580d53ce_traffic-sign-data/traffic-sign-data.zip"]

        super(StreetSignsDataProvider, self).download_if_needed(dataset_name, directory, training_file, zips)

        def get_img_and_labels(file):
            with open(file, mode='rb') as f:
                dataset = pickle.load(f)
            X, y = dataset['features'], dataset['labels']

            for i in range(X.shape[0]):
                # X_train[i] = rescale_intensity(X_train[i], in_range=(0, 255), out_range=(87, 167))
                img = X[i]
                X[i] = rescale_intensity(img, in_range=(img.min(), img.max()), out_range=(100, 190))

                # a = 3
                #c_image_np_rgb = rescale_intensity(c_image_np, in_range=(c_image_np.min(), c_image_np.max()),
                #                                   out_range=(87, 167))

            X = self.preprocess_data(X)
            all_images = np.transpose(X, (0, 3, 1, 2))
            all_labels = np.expand_dims(y, 1)

            return all_images, all_labels

        def resize_ss(images):
            # images = images / 255.
            resized_images = np.ones((images.shape[0], self.SS_CH_NUM, self.SS_DIM, self.SS_DIM), np.float32)
            for i in range(images.shape[0]):
                img = images[i][0]
                # print(img.max())
                # print(img.min())

                resized_images[i][0] = resize(img, output_shape=(self.SS_DIM, self.SS_DIM))
            return resized_images

        train_all_images, train_all_labels = get_img_and_labels(training_file)
        train_all_images = resize_ss(train_all_images)

        if self.get_test:
            test_all_images, test_all_labels = get_img_and_labels(test_file)
            test_all_images = resize_ss(test_all_images)
            return train_all_images, train_all_labels, test_all_images, test_all_labels

        return train_all_images, train_all_labels

    def get_image_size(self):
        """
        :return: returns the image size along 1 dimension (assuming square images)
        """
        return self.SS_DIM

    def get_image_channels_num(self):
        return self.SS_CH_NUM


