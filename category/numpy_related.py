import numpy as np

from category.folder_file import exist_or_create_folder


def save_array_numpy(path, array, f_mt):
    """
    Save an numpy array to a text file.
    :param path: file path involving filename, i.e., ./data/input_data.out, ./data/input_data.txt
    :param array:
    :param f_mt: i.e., '%d', '%f', '%10.5f', '%.4e' (exponential notation)
    :return:
    """
    exist_or_create_folder(path)
    np.savetxt(path, array, fmt=f_mt)


def read_array_numpy(path, d_type=float):
    """
    Load data from a text file.
    :param path: file path involving filename, i.e., ./data/input_data.out, ./data/input_data.txt
    :param d_type: float or int
    :return:
    """
    array = np.loadtxt(path, dtype=d_type)
    return array


def one_hot_encoding_numpy(array_list, size):
    """
    One Hot Encoding using numpy.
    :param array_list: i.e., [1, 2, 3]
    :param size: one hot size, i.e., 4
    :return: ndarray i.e., [[0. 1. 0. 0.]
                            [0. 0. 1. 0.]
                            [0. 0. 0. 1.]]
    """
    one_hot_array = np.eye(size)[array_list]
    return one_hot_array


def most_frequent_element_numpy(array):
    """
    find the most frequent element in a numpy array.
    :param array:
    :return:
    """
    value = np.argmax(np.bincount(array))
    return value


def normalization_zero2one_numpy(array):
    """
    normalization a numpy array to [0, 1]
    :param array:
    :return:
    """
    min_value, max_value = array.min(), array.max()
    new_array = (array - min_value) / (max_value - min_value)
    return new_array


def nan_in_array(array):
    """
    To check whether there is nan value in array.
    :param array: multi-dimensional numpy array
    :return: False -- no nan in array.
             True  -- has nan in array.
    """
    return np.isnan(np.min(array.ravel()))


def conditional_indexes(array):
    """ np.where to get conditional indexes. """
    return np.where(array > 5)


def merge_array_lists2single(mylist, list_of_list=False):
    """ Convert list of numpy arrays (or list of list) into single numpy array. """
    if list_of_list:
        # Convert 'list of list' to 'list of numpy array'.
        mylist = list(map(np.array, mylist))

    # # 1. np.array(LIST)
    # # This method only works for the specific case of
    # # vertical stacking of 1-D arrays with same length such as demo_list_1.
    #
    # # Works for demo_list_1.
    # return_array = np.array(mylist)

    # # 2. np.concatenate(LIST, axis=0)
    # # This method is general, but you'll have to expand dimensionality
    # # for 1-D arrays.
    #
    # # Works for demo_list_2, demo_list_3 and
    # # demo_list_1 (have to expand dimensionality manually).
    # return_array = np.concatenate(mylist, axis=0)

    # # 3. np.stack(LIST, axis=0)
    # # This method is general, but it expands dimensionality for
    # # 1-D arrays automatically.
    #
    # # Works for demo_list_1.
    # return_array = np.stack(mylist, axis=0)

    # # 4. np.vstack(LIST)
    # # This method is the Best.
    #
    # # Works for demo_list_1, demo_list_2, demo_list_3.
    return_array = np.vstack(mylist)

    return return_array


def count_items(array):
    """ Count the occurrence of certain item in an ndarray. """
    unique, counts = np.unique(array, return_counts=True)
    return dict(zip(unique, counts))


def main():
    # # ---- Transfer an bin-array to int(scalar). ---- #
    # print(nan_in_array(np.array([0, 1, 1, 0])))
    # # ------- Get indexes of whose value > 5. ------- #
    # print(conditional_indexes(np.arange(10)))
    # ---------- Convert to a single list. ---------- #
    # # demo_list_1's shape: [(5,), (5,), (5,)]
    # demo_list_1 = [np.array([1, 1, 1, 1, 1]),
    #                np.array([2, 2, 2, 2, 2]),
    #                np.array([3, 3, 3, 3, 3])]
    # # demo_list_2's shape: [(1, 5), (1, 5), (1, 5)]
    # demo_list_2 = [np.array([1, 1, 1, 1, 1])[np.newaxis, :],
    #                np.array([2, 2, 2, 2, 2])[np.newaxis, :],
    #                np.array([3, 3, 3, 3, 3])[np.newaxis, :]]
    # # demo_list_3's shape: [(1, 5), (2, 5), (3, 5)]
    # demo_list_3 = [np.array([1, 1, 1, 1, 1])[np.newaxis, :],
    #                np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2]).reshape((2, 5)),
    #                np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]).reshape((3, 5))]
    # merge_array = merge_array_lists2single(demo_list_1, list_of_list=False)
    # print(merge_array.shape)
    # demo_list_4 = [[1, 1, 1, 1, 1],
    #                [2, 2, 2, 2, 2],
    #                [3, 3, 3, 3, 3]]
    # merge_array = merge_array_lists2single(demo_list_4, list_of_list=True)
    # print(merge_array.shape)
    arr = np.array([0, 1, 2, 3, 0, 2, 5, 8, 3, 2, 0])
    print(count_items(arr))


if __name__ == '__main__':
    main()
