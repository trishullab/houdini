import numpy as np


class NumpyDataSetIterator(object):
    """Generic data provider."""

    def __init__(self, inputs, targets, batch_size, max_num_batches=-1,
                 shuffle_order=True, rng=None):
        """Create a new data provider object.
        Args:
            inputs (ndarray): Array of data input features of shape
                (num_data, input_dim).
            targets (ndarray): Array of data output targets of shape
                (num_data, output_dim) or (num_data,) if output_dim == 1.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        self.inputs = inputs
        self.targets = targets
        if batch_size < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = batch_size
        if max_num_batches == 0 or max_num_batches < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = max_num_batches
        self._update_num_batches()
        self.shuffle_order = shuffle_order
        self._current_order = np.arange(inputs.shape[0])
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng
        self.new_epoch()

    @property
    def batch_size(self):
        """Number of data points to include in each batch."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = value
        self._update_num_batches()

    @property
    def num_datapoints(self):
        num_full_batches = self.inputs.shape[0] // self.batch_size
        if self.inputs.shape[0] % self.batch_size == 0:
            return num_full_batches * self.batch_size
        else:
            return num_full_batches * self.batch_size + self.inputs.shape[0] % self.batch_size

    @property
    def max_num_batches(self):
        """Maximum number of batches to iterate over in an epoch."""
        return self._max_num_batches

    @max_num_batches.setter
    def max_num_batches(self, value):
        if value == 0 or value < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = value
        self._update_num_batches()

    def _update_num_batches(self):
        """Updates number of batches to iterate over."""
        # maximum possible number of batches is equal to number of whole times
        # batch_size divides in to the number of data points which can be
        # found using integer division
        possible_num_batches = self.inputs.shape[0] // self.batch_size + (1 if self.inputs.shape[0] % self.batch_size != 0 else 0)
        if self.max_num_batches == -1:
            self.num_batches = possible_num_batches
        else:
            self.num_batches = min(self.max_num_batches, possible_num_batches)

    def __iter__(self):
        """Implements Python iterator interface.
        This should return an object implementing a `next` method which steps
        through a sequence returning one element at a time and raising
        `StopIteration` when at the end of the sequence. Here the object
        returned is the DataProvider itself.
        """
        return self

    def new_epoch(self):
        """Starts a new epoch (pass through data), possibly shuffling first."""
        self._curr_batch = 0
        if self.shuffle_order:
            self.shuffle()

    def __next__(self):
        return self.next()

    def reset(self):
        """Resets the provider to the initial state."""
        inv_perm = np.argsort(self._current_order)
        self._current_order = self._current_order[inv_perm]
        self.inputs = self.inputs[inv_perm]
        self.targets = self.targets[inv_perm]
        self.new_epoch()

    def shuffle(self):
        """Randomly shuffles order of data."""
        perm = self.rng.permutation(self.inputs.shape[0])
        self._current_order = self._current_order[perm]
        self.inputs = self.inputs[perm]
        self.targets = self.targets[perm]

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        if self._curr_batch + 1 > self.num_batches:
            # no more batches in current iteration through data set so start
            # new epoch ready for another pass and indicate iteration is at end
            self.new_epoch()
            raise StopIteration()
        # create an index slice corresponding to current batch number
        batch_slice = slice(self._curr_batch * self.batch_size,
                            (self._curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self._curr_batch += 1
        return inputs_batch, targets_batch


class ImageGraphDataGenerator(NumpyDataSetIterator):
    """"""

    def __init__(self, inputs, labels, batch_size, grid_size, data_provider, img_dictionary,
                 value_to_class_dict, img_class_start, img_class_goal):
        """
        Storing all mazes with images is very inefficient, as a lot of images would be repeatedly stored in memory.
        This class adds images to a maze, returned from super().next().
        """
        self.grid_size = grid_size
        self.data_provider = data_provider
        self.img_dictionary = img_dictionary
        self.val_to_class_dict = value_to_class_dict

        self.img_class_start = img_class_start
        self.img_class_goal = img_class_goal

        # load data from compressed numpy file
        # loaded = np.load(data_path)
        # inputs, targets = loaded['inputs'], loaded['targets']
        # inputs = inputs.astype(np.float32)
        # pass the loaded data to the parent class __init__
        super(ImageGraphDataGenerator, self).__init__(inputs, labels, batch_size, -1, True, None)

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        maze, maze_labels = super(ImageGraphDataGenerator, self).next()

        start = [0, 0]
        goal = [self.grid_size-1, self.grid_size-1]

        img_dim = self.data_provider.get_image_size()
        img_ch = self.data_provider.get_image_channels_num()
        maze_w_imgs = np.zeros(shape=list(maze.shape) + [img_ch, img_dim, img_dim], dtype=np.float32)

        for b in range(maze.shape[0]):
            for r in range(maze.shape[1]):
                for c in range(maze.shape[2]):
                    if r == start[0] and c == start[1]:
                        c_img_class = self.img_class_start
                    elif r == goal[0] and c == goal[1]:
                        c_img_class = self.img_class_goal
                    else:
                        c_maze_value = maze[b, r, c]
                        c_img_class = self.val_to_class_dict[c_maze_value]
                    c_rndm_img = self.data_provider.get_random_image(c_img_class, self.img_dictionary)
                    maze_w_imgs[b, r, c] = c_rndm_img

        # print(maze_w_imgs.shape)

        return maze_w_imgs, maze_labels


if __name__ == '__main__':
    inputs = np.array([1, 8, 3, 4, 5]).reshape(-1, 1)
    labels = np.array([1, 8, 3, 4, 5]).reshape(-1, 1)

    # inputs = inputs[:1]
    # labels = labels[:1]


    # inputs = np.ones((5, 1), np.float32)
    # labels = np.zeros((5, 1), np.float32)
    # inputs = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
    # labels = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])

    print(inputs)
    g = NumpyDataSetIterator(inputs, labels, batch_size=2)
    print(inputs)

    # g = ImageGraphDataGenerator(inputs, labels, batch_size=2)


    for e in range(2):
        for i, l in g:
            print(i)
            print(l)
            print("-------------------------")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    """"""
    print(inputs)
    print(g.inputs)
