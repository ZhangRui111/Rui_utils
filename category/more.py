import numpy as np


def binary_array_to_int(array):
    """
    Transfer an bin-array to int(scalar).
    :param array: ndarray or list, such as [0, 1, 1, 0].
    :return: int(scalar), such as 6.
    """
    input = array
    out = 0
    for bit in input:
        out = (out << 1) | bit
    return out


def variable_exist_before(variable):
    """
    To check whether a variable exists before now.
    :param variable:
    :return:
    """
    if variable not in globals() and variable not in locals():
        exist_before = False
    else:
        exist_before = True
    return exist_before


# def main():
#     # ---- Transfer an bin-array to int(scalar). ---- #
#     print(binary_array_to_int(np.array([0, 1, 1, 0])))
#     print(binary_array_to_int([0, 1, 1, 0]))
#
#
# if __name__ == '__main__':
#     main()
