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
