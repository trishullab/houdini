import numpy as np
import collections
import sys
import os
import random

maze_direcory = "Data/Mazes/"
class MazeGenerator:

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
    def iterate_thourgh_mazes(c_maze, c_row, c_col, grid_size, elements, chance):

        if c_row == grid_size-1 and c_col == grid_size-1:
            if random.uniform(0, 1) < chance:
                c_maze[c_row][c_col] = 1
                yield c_maze
            return

        if c_col == grid_size-1:
            next_col = 0
            next_row = c_row + 1
        else:
            next_col = c_col + 1
            next_row = c_row

        for el in elements:
            c_maze[c_row][c_col] = el
            yield from MazeGenerator.iterate_thourgh_mazes(c_maze, next_row, next_col, grid_size, elements, chance)


    @staticmethod
    def generate_data_tr_val_test(num_sample_tr, num_sample_val, grid_size, elements,
                                  only_last_shortest_path=True):
        num_elements = elements.__len__()
        num_unique_mazes = num_elements**(grid_size*grid_size-2) # start/goal are constant.
        num_items_to_generate = num_sample_tr+num_sample_val+num_sample_val # num_sample_test = num_sample_val
        assert (num_unique_mazes >= num_sample_tr + num_sample_val + num_sample_val)

        start = (0, 0)
        goal = [grid_size - 1, grid_size - 1]

        filename = "maze_{}x{}_{}els_{}items".format(grid_size, grid_size, elements.__len__(), num_items_to_generate)
        filename_labels = filename + "_labels"
        filepath = maze_direcory+"/{}.npy".format(filename)
        filepath_labels = maze_direcory + "/{}.npy".format(filename_labels)

        # if the mazes are stored, load them, else compute and store
        if os.path.exists(maze_direcory) and os.path.exists(filepath) and os.path.exists(filepath_labels):
            mazes = np.load(filepath)
            mazes_labels = np.load(filepath_labels)
        else:
            max_numbers_to_generate = num_items_to_generate + (num_items_to_generate // 5)  # generate 20% extra
            keep_chance = float(max_numbers_to_generate) / float(num_unique_mazes)

            mazes = np.zeros(shape=(max_numbers_to_generate, grid_size, grid_size), dtype=np.float32)
            if only_last_shortest_path:
                mazes_labels = np.zeros(shape=(max_numbers_to_generate, 1), dtype=np.float32)
            else:
                mazes_labels = np.zeros_like(mazes)

            # adding 1000 to get a buffer and make sure we generate enough examples
            # print(num_items_to_generate)
            # print(num_unique_mazes)

            # print("keep_chance={}".format(keep_chance))
            # print("expected num items={}".format(keep_chance*num_unique_mazes))

            if not os.path.exists(maze_direcory):
                os.makedirs(maze_direcory)

            c_maze = np.ones((grid_size, grid_size), np.float32)
            c_maze[start[0]][start[1]] = 1.
            c_maze[goal[0]][goal[1]] = 1.

            c_idx = 0
            for m in MazeGenerator.iterate_thourgh_mazes(c_maze, 0, 1, grid_size, elements, keep_chance):
                exists_shortest_path, shortest_path_len, shortest_paths = MazeGenerator.shortest_path_bfm(m, grid_size, start)
                if exists_shortest_path:
                    mazes[c_idx] = m
                    if only_last_shortest_path:
                        mazes_labels[c_idx] = shortest_path_len
                    else:
                        mazes_labels[c_idx] = shortest_paths
                    c_idx += 1

                    if c_idx % 10000 == 0:
                        print("{}/{}".format(c_idx, num_unique_mazes))

                    if c_idx >= max_numbers_to_generate:
                        break
                # print(m)

            num_items_generated = c_idx
            assert(num_items_generated >= num_items_to_generate)
            #remove the excessive 0 entries
            mazes = mazes[:num_items_generated]
            mazes_labels = mazes_labels[:num_items_generated]
            # shuffle
            MazeGenerator.shuffle_in_unison(mazes, mazes_labels, None)
            # pick the first num_items_to_generate items
            mazes = mazes[:num_items_to_generate]
            mazes_labels = mazes_labels[:num_items_to_generate]

            np.save(filepath, mazes)
            np.save(filepath_labels, mazes_labels)

        # Presumably, the numpy array should've been shuffled before saving, so no need to shuffle again.
        # MazeGenerator.shuffle_in_unison(mazes, mazes_labels, None)

        """
        for i in range(mazes.shape[0]):
            print(mazes[i])
            print(mazes_labels[i])
            print("++++++++++++")
        """

        tr_mazes = mazes[:num_sample_tr]
        tr_mazes_labels = mazes_labels[:num_sample_tr]

        val_mazes = mazes[num_sample_tr:num_sample_tr + num_sample_val]
        val_mazes_labels = mazes_labels[num_sample_tr:num_sample_tr + num_sample_val]

        test_mazes = mazes[num_items_to_generate-num_sample_val:]
        test_mazes_labels = mazes_labels[num_items_to_generate - num_sample_val:]

        return tr_mazes, tr_mazes_labels, val_mazes, val_mazes_labels, test_mazes, test_mazes_labels

    @staticmethod
    def generate_data_tr_val_test_faster(num_sample_tr, num_sample_val, grid_size, elements,
                                         only_last_shortest_path=True):
        num_elements = elements.__len__()
        num_unique_mazes = num_elements ** (grid_size * grid_size - 2)  # start/goal are constant.
        # num_items_to_generate = num_sample_tr + num_sample_val
        num_items_to_generate = num_sample_tr + num_sample_val + num_sample_val  # num_sample_test = num_sample_val

        assert (num_unique_mazes >= num_sample_tr + num_sample_val)

        start = (0, 0)
        goal = [grid_size - 1, grid_size - 1]

        filename = "maze_{}x{}_{}els_{}items".format(grid_size, grid_size, elements.__len__(), num_items_to_generate)
        filename_labels = filename + "_labels"
        filepath = maze_direcory + "/{}.npy".format(filename)
        filepath_labels = maze_direcory + "/{}.npy".format(filename_labels)

        # if the mazes are stored, load them, else compute and store
        if os.path.exists(maze_direcory) and os.path.exists(filepath) and os.path.exists(filepath_labels):
            mazes = np.load(filepath)
            mazes_labels = np.load(filepath_labels)
        else:
            max_numbers_to_generate = num_items_to_generate*3  # generate 3 times as much

            while True:
                print("generating random mazes")
                rndm_mazes = np.random.choice(elements, (max_numbers_to_generate, grid_size, grid_size))
                rndm_mazes[:, 0, 0] = 1.
                rndm_mazes[:, grid_size - 1, grid_size - 1] = 1.
                print("finding the unique mazes")
                unique_mazes = np.unique(rndm_mazes, axis=0)
                if unique_mazes.shape[0] >= num_items_to_generate:
                    break

            np.random.shuffle(unique_mazes)
            mazes = unique_mazes[:num_items_to_generate]
            if only_last_shortest_path:
                mazes_labels = np.zeros((num_items_to_generate, 1), np.float32)
            else:
                mazes_labels = np.zeros_like(mazes, np.float32)

            if not os.path.exists(maze_direcory):
                os.makedirs(maze_direcory)

            for m_idx in range(mazes.shape[0]):
                c_maze = mazes[m_idx]
                exists_shortest_path, shortest_path_len, shortest_paths = MazeGenerator.shortest_path_bfm(c_maze, grid_size, start)
                if only_last_shortest_path:
                    mazes_labels[m_idx] = shortest_path_len
                else:
                    mazes_labels[m_idx] = shortest_paths
                if m_idx % 10000 == 0:
                    print("{}/{}".format(m_idx, num_unique_mazes))

            np.save(filepath, mazes)
            np.save(filepath_labels, mazes_labels)

        # Presumably, the numpy array should've been shuffled before saving, so no need to shuffle again.
        # MazeGenerator.shuffle_in_unison(mazes, mazes_labels, None)

        """
        for i in range(mazes.shape[0]):
            print(mazes[i])
            print(mazes_labels[i])
            print("++++++++++++")
        """

        #tr_mazes = mazes[:num_sample_tr]
        #tr_mazes_labels = mazes_labels[:num_sample_tr]
        #val_mazes = mazes[num_items_to_generate - num_sample_val:]
        #val_mazes_labels = mazes_labels[num_items_to_generate - num_sample_val:]

        tr_mazes = mazes[:num_sample_tr]
        tr_mazes_labels = mazes_labels[:num_sample_tr]

        val_mazes = mazes[num_sample_tr:num_sample_tr + num_sample_val]
        val_mazes_labels = mazes_labels[num_sample_tr:num_sample_tr + num_sample_val]

        test_mazes = mazes[num_items_to_generate - num_sample_val:]
        test_mazes_labels = mazes_labels[num_items_to_generate - num_sample_val:]

        return tr_mazes, tr_mazes_labels, val_mazes, val_mazes_labels, test_mazes, test_mazes_labels

    @staticmethod
    def generate_data(num_samples, grid_size, elements):
        """Set grid params
        exaple elements = (10./3., 10./5., 10./7.)
        """

        start = (0, 0)
        goal = [grid_size - 1, grid_size - 1]

        dataset_input = np.zeros(shape=(num_samples, grid_size, grid_size), dtype=np.float32)
        dataset_output = np.zeros(shape=(num_samples, 1), dtype=np.float32)

        i = 0
        while i < num_samples:
            grid = MazeGenerator.generate_maze(grid_size, start, goal, elements)
            exists_shortest_path, shortest_path_len, _ = MazeGenerator.shortest_path_bfm(grid, grid_size, start)
            if exists_shortest_path:
                dataset_input[i] = grid
                dataset_output[i] = shortest_path_len
                i += 1

                if i % 10000 == 0:
                    print("{}/{}".format(i, num_samples))

        # dataset_input = np.expand_dims(dataset_input, axis=1)
        # print("number of unique data points = {}".format(grid_size))
        return dataset_input, dataset_output, start, goal

    @staticmethod
    def generate_maze(grid_size, start, goal, elements):
        # if we want to make blocks, we can use the previous code to generate a {0, 1} maze
        # afterwards, we can simply use the mask=(maze == 0), and do maze[mask] = rndm_choice_array[mask]
        maze = np.random.choice(elements, (grid_size, grid_size))
        maze[start[0]][start[1]] = 1.
        maze[goal[0]][goal[1]] = 1.
        return maze

    @staticmethod
    def update_val_at_pos(grid, grid_size, values, row, col):
        min_val = values[row][col]

        c_row = row - 1
        c_col = col
        for c_row, c_col in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]:
            if 0 <= c_row < grid_size and 0 <= c_col < grid_size:
                c_val = values[c_row, c_col] + grid[row][col]
                if c_val < min_val:
                    min_val = c_val

        values[row][col] = min_val

    @staticmethod
    def update_val_at_pos_look_at_old_values_only(grid, grid_size, old_values, values, row, col):
        """This way the function doesn't have access to the newest updates attained during the convolution."""
        min_val = values[row][col]

        c_row = row - 1
        c_col = col
        for c_row, c_col in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]:
            if 0 <= c_row < grid_size and 0 <= c_col < grid_size:
                c_val = old_values[c_row, c_col] + grid[row][col]
                if c_val < min_val:
                    min_val = c_val

        values[row][col] = min_val

    @staticmethod
    def shortest_path_bfm_laovo(grid, grid_size, start):  # bfm= bellman ford modified
        values = np.ones_like(grid) * sys.float_info.max - 2
        values[start[0], start[1]] = 1

        for i in range(6):  # range(grid.size - 1):
            old_values = np.copy(values)
            for row in range(grid_size):
                for col in range(grid_size):
                    if row == start[0] and col == start[1]:
                        continue
                    MazeGenerator.update_val_at_pos_look_at_old_values_only(grid, grid_size, old_values, values, row, col)

        path_length_to_goal = values[-1, -1]
        return path_length_to_goal < 100000., path_length_to_goal, values

    @staticmethod
    def shortest_path_bfm(grid, grid_size, start):  # bfm= bellman ford modified
        values = np.ones_like(grid) * sys.float_info.max - 2
        values[start[0], start[1]] = 1

        for i in range(grid.size - 1):
            for row in range(grid_size):
                for col in range(grid_size):
                    if row == start[0] and col == start[1]:
                        continue
                    MazeGenerator.update_val_at_pos(grid, grid_size, values, row, col)

        path_length_to_goal = values[-1, -1]
        return path_length_to_goal < 100000., path_length_to_goal, values


if __name__ == '__main__':
    grid_size = 4
    start = (0, 0)
    goal = (grid_size-1, grid_size-1)
    diff = 0

    num_random_mazes = 100000
    for i in range(num_random_mazes):
        maze = MazeGenerator.generate_maze(grid_size, start, goal, elements=(1, 3, 5, 7, 9))
        a, b, c = MazeGenerator.shortest_path_bfm(maze, grid_size, start)
        d, e, f = MazeGenerator.shortest_path_bfm_laovo(maze, grid_size, start)

        if (c == f).all():
            pass
            #print("match")
        else:
            print("hm, difference")
            diff += 1

            """
            print("maze:")
            print(maze)
            print("values:")
            print(c)
            print(f)
            print("____________________________-")
            """
    print("different items: {}/{}".format(diff, num_random_mazes))
