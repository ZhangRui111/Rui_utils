import csv
import numpy as np

from category.folder_file import exist_or_create_folder


def save_txt(path, data, if_overwrite):
    """
    Save to a file.
    :param path: file path involving filename, i.e., ./data/input_data.txt
    :param data:
    :param if_overwrite: Whether overwrite the original content.
    :return:
    """
    exist_or_create_folder(path)
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
