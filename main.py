import numpy as np
import sys
import time

sys.path.append("category/")
import folder_file as ff
import more as more
import matplotlib_pyplot as mp
import numpy_related as npr
import save_and_read as sr

# Your path (i.e. the list of directories Python goes through to search for modules and files)
# is stored in the path attribute of the sys module. Since path is a list, you can use the
# append method to add new directories to the path.
sys.path.append("test/")
from subfolder1.module1 import func1
from subfolder2.module2 import func2


def main(*args):
    local_args = args
    print([local_args[0][i] for i in range(len(local_args[0]))])
    # ---------------- time recoder --------------- #
    start_time = time.time()
    # ------ add new directories to the path ------ #
    func1()
    func2()
    # ---------- Save and Read csv file ----------- #
    print("Write and Read csv file")
    sr.save_csv("./test/test.csv", [[1, 2, 3], [2, 5, 9]], ",", True)
    content = sr.read_csv("./test/test.csv", ",")
    print(content)
    # ------------- One Hot Encoding -------------- #
    print("One Hot Encoding")
    in_array = np.array([1, 2, 3])
    one_hot_array = npr.one_hot_encoding_numpy(in_array, in_array.max() + 1)
    print(one_hot_array)
    # ------------ exist or create path ----------- #
    print("exist or create path")
    result = sr.exist_or_create_folder("./logs/")
    print("Create: ", result)
    # ------- find the most frequent element ------ #
    print("find the most frequent element")
    value = npr.most_frequent_element_numpy(np.array([1, 2, 3, 4, 2, 3, 1, 2, 2, 9]))
    print(value)
    # ----------- normalization to [0, 1] --------- #
    print("normalization to [0, 1]")
    old_array = np.array([1, 2, 3, 4, 2, 3, 1, 2, 2, 9])
    print(old_array)
    new_array = npr.normalization_zero2one_numpy(old_array)
    print(new_array)
    # ---------------- time recoder -------------- #
    for i in range(5):
        time.sleep(1)
    print("Time recoder: {0}".format(time.time() - start_time))
    # # ---- Save and Read ndarray from txt file. ---- #
    # print("Save and Read ndarray from txt file.")
    # array_1 = np.array([0, 1, 2, 3, 4, 5], dtype=int)
    # array_2 = np.array([0.2, 3.56, 24.88881253, 1.258, 0.3698745, 1.25, 10.])
    # array_3 = np.array([0.0000001, 0.000000025, 0.3666856, 0.12500004])
    # sr.save_array_numpy("./test/array_1.out", array_1, "%d")
    # sr.save_array_numpy("./test/array_21.out", array_2, "%f")
    # sr.save_array_numpy("./test/array_22.out", array_2, "%3.10f")
    # sr.save_array_numpy("./test/array_3.out", array_3, "%e")
    # array_1_1 = sr.read_array_numpy("./test/array_1.out", d_type=int)
    # array_2_21 = sr.read_array_numpy("./test/array_21.out", d_type=float)
    # array_2_22 = sr.read_array_numpy("./test/array_22.out", d_type=float)
    # array_3_1 = sr.read_array_numpy("./test/array_3.out", d_type=float)
    # print(array_1_1)
    # print(array_2_21)
    # print(array_2_22)
    # print(array_3_1)
    # # ------------- Save and Read file ------------ #
    # sr.save_txt("./test/test.txt", ['apple', 'banana', 'pear'], True)
    # content = sr.read_txt("./test/test.txt")
    # print(content)
    # # -------------------- plot ------------------- #
    # print("plot numpy array in segment mean value")
    # data = np.arange(5000)
    # mp.read_output_plot(data)
    # ---------------- class attribute -------------- #
    # print(hasattr(self, 'label'))  # in-class function
    # dqn = DQN()
    # print(hasattr(dqn, 'label'))  # other function


if __name__ == "__main__":
    """
    Receive args from command line or bat/cmd files.
    input: python main.py zhang rui
    output: ['main.py', 'zhang', 'rui']
    """
    argv = sys.argv
    # # ----- catch KeyboardInterrupt [ctrl + C] ---- #
    try:
        print("do normal")
        main(argv)
    except KeyboardInterrupt:
        print("catch KeyboardInterrupt and do something!")
