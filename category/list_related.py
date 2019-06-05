def remove_duplicates(mylist):
    """ Remove Duplicates From a Python List. """
    return_list = list(dict.fromkeys(mylist))
    return return_list


def convert_items_type(mylist, token):
    """ Convert all items in a list to int/float/str type. """
    if token == 'int':
        return_list = list(map(int, mylist))
        # return_list = map(int, mylist)  # (in python 2.X)
    elif token == 'float':
        return_list = list(map(float, mylist))
    elif token == 'str':
        return_list = list(map(str, mylist))
    else:
        raise ValueError(
            "The second parameter 'token' must be 'int', 'float' or 'str'.")
    return return_list

######################################
# map(fun, iter):
# map() function returns a list of the results after applying the given
# function to each item of a given iterable (list, tuple etc.)

# ---------------------------------- #
# # Return double of n
# def addition(n):
#     return n + n
#
# # We double all numbers using map()
# numbers = (1, 2, 3, 4)
# result = map(addition, numbers)
# print(list(result))
#
# # Output: {2, 4, 6, 8}
######################################


# def main():
    # # ---- Remove Duplicates From a Python List. ---- #
    # my_list = [1, 2, 3, 3, 4, 5, 6, 6, 4, 7]
    # my_list = ['a', 'b', 'c', 'd', 'e', 'a', 'b', 'a']
    # print(remove_duplicates(my_list))
    # # ---- Remove Duplicates From a Python List. ---- #
    # my_list = [1, 2, 3, 3, 4, 5, 6, 6, 4, 7]
    # my_list_str = convert_items_type(my_list, 'str')
    # my_list_int = convert_items_type(my_list_str, 'int')
    # my_list_float = convert_items_type(my_list_str, 'float')
    # print(my_list_str)
    # print(my_list_int)
    # print(my_list_float)


# if __name__ == '__main__':
#     main()
