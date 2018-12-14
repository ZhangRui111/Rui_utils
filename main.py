import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time


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


def save_txt(path, data, if_overwrite):
    """
    Save to a file.
    :param path: file path involving filename, i.e., ./data/input_data.txt
    :param data:
    :param if_overwrite: Whether overwrite the original content.
    :return:
    """
    exist_or_create_path(path)
    if if_overwrite is True:
        with open(path, 'w') as f:
            f.write(str(data))
            # f.writelines(list)
    else:
        with open(path, 'a') as f:
            f.write(data)
            # f.writelines(list)


def read_txt(path):
    """
    Read from a file.
    :param path: file path involving filename, i.e., ./data/input_data.txt
    :return:
    """
    try:
        with open(path, 'r') as f:
            # The read() method just outputs the entire file if number of bytes are not given in the argument.
            content = f.read()
            # # readline(n) outputs at most n bytes of a single line of a file. It does not read more than one line.
            # content = f.readline()
            return content
    except IOError:
        print("File not found or path is incorrect")
    finally:
        print("exit")


def save_csv(path, data, delimiter=',', if_overwrite=False):
    """
    Save to CSV file.
    :param path: file path involving filename, i.e., ./data/input_data.csv
    :param data: a list, i.e., [[1, 2, 3], ['Morning', 'Evening', 'Afternoon']]
    :param delimiter: delimiter to separate different elements.
    :param if_overwrite: Whether overwrite the original content.
    :return:
    """
    exist_or_create_path(path)

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


def save_array_numpy(path, array, f_mt):
    """
    Save an numpy array to a text file.
    :param path: file path involving filename, i.e., ./data/input_data.out, ./data/input_data.txt
    :param array:
    :param f_mt: i.e., '%d', '%f', '%10.5f', '%.4e' (exponential notation)
    :return:
    """
    exist_or_create_path(path)
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


def read_output_plot(data, savepath=None, if_close_figure=True):
    """
    plot numpy array in segment mean value.
    :param data: numpy array.
    :param savepath:  figure save path.
    :param if_close_figure:  whether plt.close()
    :return:
    """
    data_plot = []
    length = data.size
    interval = 250
    size = int(length / interval)
    for i in range(size):
        start = i * interval
        end = (i + 1) * interval
        segment = data[start:end]
        data_plot.append(np.mean(segment))
    x_axis_data = np.arange(0, length, interval)

    plt.plot(x_axis_data, np.asarray(data_plot), label='label')
    plt.title('title')  # plot figure title
    plt.xlabel('xlabel')  # plot figure's x axis name.
    plt.ylabel('ylabel')  # plot figure's y axis name.
    y_axis_ticks = [0, 1000, 2000, 3000, 4000, 5000]  # range of y axis
    plt.yticks(y_axis_ticks)  # set y axis's ticks
    for items in y_axis_ticks:  # plot some lines that vertical to y axis.
        plt.hlines(items, x_axis_data.min(), x_axis_data.max(), colors="#D3D3D3", linestyles="dashed")
    plt.legend(loc='best')
    if savepath is not None:
        plt.savefig(savepath + 'data.png')  # save figures.
    plt.show()
    if if_close_figure is True:
        plt.close()  # if not close figure, then all plot will be drawn in the same figure.


def main(*args):
    local_args = args
    print([local_args[0][i] for i in range(len(local_args[0]))])
    # # ---------------- time recoder --------------- #
    # start_time = time.time()
    # # ---------- Save and Read csv file ---------- #
    # print('Write and Read csv file')
    # save_csv('./test/test.csv', [[1, 2, 3], [2, 5, 9]], ',', True)
    # content = read_csv('./test/test.csv', ',')
    # print(content)
    # # ------------- One Hot Encoding -------------- #
    # print('One Hot Encoding')
    # in_array = np.array([1, 2, 3])
    # one_hot_array = one_hot_encoding_numpy(in_array, in_array.max()+1)
    # print(one_hot_array)
    # # ------------ exist or create path ----------- #
    # print('exist or create path')
    # result = exist_or_create_path('./logs/')
    # print('Create: ', result)
    # # ------- find the most frequent element ------ #
    # print('find the most frequent element')
    # value = most_frequent_element_numpy(np.array([1, 2, 3, 4, 2, 3, 1, 2, 2, 9]))
    # print(value)
    # # ----------- normalization to [0, 1] --------- #
    # print('normalization to [0, 1]')
    # old_array = np.array([1, 2, 3, 4, 2, 3, 1, 2, 2, 9])
    # print(old_array)
    # new_array = normalization_zero2one_numpy(old_array)
    # print(new_array)
    # # ---------------- time recoder -------------- #
    # for i in range(5):
    #     time.sleep(1)
    # print('Time recoder: {0}'.format(time.time()-start_time))
    # # ---- Save and Read ndarray from txt file. ---- #
    # print('Save and Read ndarray from txt file.')
    # array_1 = np.array([0, 1, 2, 3, 4, 5], dtype=int)
    # array_2 = np.array([0.2, 3.56, 24.88881253, 1.258, 0.3698745, 1.25, 10.])
    # array_3 = np.array([0.0000001, 0.000000025, 0.3666856, 0.12500004])
    # save_array_numpy('./test/array_1.out', array_1, '%d')
    # save_array_numpy('./test/array_21.out', array_2, '%f')
    # save_array_numpy('./test/array_22.out', array_2, '%3.10f')
    # save_array_numpy('./test/array_3.out', array_3, '%e')
    # array_1_1 = read_array_numpy('./test/array_1.out', d_type=int)
    # array_2_21 = read_array_numpy('./test/array_21.out', d_type=float)
    # array_2_22 = read_array_numpy('./test/array_22.out', d_type=float)
    # array_3_1 = read_array_numpy('./test/array_3.out', d_type=float)
    # print(array_1_1)
    # print(array_2_21)
    # print(array_2_22)
    # print(array_3_1)
    # # ------------- Save and Read file ------------ #
    # save_txt('./test/test.txt', ['apple', 'banana', 'pear'], True)
    # content = read_txt('./test/test.txt')
    # print(content)
    # # -------------------- plot ------------------- #
    # print('plot numpy array in segment mean value')
    # data = np.arange(5000)
    # read_output_plot(data)


if __name__ == '__main__':
    """
    Receive args from command line or bat/cmd files.
    input: python main.py zhang rui
    output: ['main.py', 'zhang', 'rui']
    """
    argv = sys.argv
    main(argv)
