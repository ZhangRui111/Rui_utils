import csv
import numpy as np
import os
import time


def read_csv(path, delimiter=','):
    """
    Read CSV file.
    :param path: file path involving filename, i.e., ./data/input_data.csv
    :param delimiter: delimiter to separate different elements.
    :return: a list involving the csv file's content.
    """
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        line_count = 0
        test_data = []
        for row in csv_reader:
            test_data.append(row)
            # ----------------------------------------------- #
            # # Below is how to read float value from row.
            # if line_count == 0:
            #     line_count += 1
            # else:
            #     test_data.append([float(row[i]) for i in range(4097)])
            #     line_count += 1
            # ----------------------------------------------- #
        return test_data


def write_csv(path, data, delimiter=',', if_overwrite=False):
    """
    Write to CSV file.
    :param path: file path involving filename, i.e., ./data/input_data.csv
    :param data: a list, i.e., [[1, 2, 3], ['Morning', 'Evening', 'Afternoon']]
    :param delimiter: delimiter to separate different elements.
    :param if_overwrite: Whether overwrite the original content.
    :return:
    """
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError:
            pass

    if if_overwrite is True:
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=delimiter)
            for line in data:
                writer.writerow(line)
    else:
        with open(path, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=delimiter)
            for line in data:
                writer.writerow(line)


def exist_or_create_path(path):
    """
    Check whether a path exists, if not, then create this path.
    Usually embedded in writing operation.
    :param path: i.e., './logs/log.txt' or './logs/'
    :return: create --> True
    """
    flag = False
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
            flag = True
        except OSError:
            pass
    return flag


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
    find the most frequent element in a numpy ndarray.
    :param array:
    :return:
    """
    value = np.argmax(np.bincount(array))
    return value


def normalization_zero2one_numpy(array):
    """
    normalization a ndarray to [0, 1]
    :param array:
    :return:
    """
    min_value, max_value = array.min(), array.max()
    new_array = (array - min_value) / (max_value - min_value)
    return new_array


def main():
    # # ---------------- time recoder --------------- #
    start_time = time.time()
    # ---------- Write and Read csv file ---------- #
    print('Write and Read csv file')
    write_csv('./test/test.csv', [[1, 2, 3], [2, 5, 9]], ',', True)
    content = read_csv('./test/test.csv', ',')
    print(content)
    # ------------- One Hot Encoding -------------- #
    print('One Hot Encoding')
    in_array = np.array([1, 2, 3])
    one_hot_array = one_hot_encoding_numpy(in_array, in_array.max()+1)
    print(one_hot_array)
    # ------------ exist or create path ----------- #
    print('exist or create path')
    result = exist_or_create_path('./logs/')
    print('Create: ', result)
    # ------- find the most frequent element ------ #
    print('find the most frequent element')
    value = most_frequent_element_numpy(np.array([1, 2, 3, 4, 2, 3, 1, 2, 2, 9]))
    print(value)
    # ----------- normalization to [0, 1] --------- #
    print('normalization to [0, 1]')
    old_array = np.array([1, 2, 3, 4, 2, 3, 1, 2, 2, 9])
    print(old_array)
    new_array = normalization_zero2one_numpy(old_array)
    print(new_array)
    # # ---------------- time recoder -------------- #
    for i in range(5):
        time.sleep(1)
    print('Time recoder: {0}'.format(time.time()-start_time))


if __name__ == '__main__':
    main()
