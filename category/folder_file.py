import os


def exist_or_create_folder(path_plus_name):
    """
    Check whether a path exists, if not, then create this path.
    Usually embedded in writing operation.
    :param path: i.e., './logs/log.txt' or './logs/'
    :return: create --> True
    """
    flag = False
    if not os.path.exists(os.path.dirname(path_plus_name)):
        try:
            os.makedirs(os.path.dirname(path_plus_name))
            flag = True
        except OSError:
            pass
    return flag


def exist_folder(path):
    pass


def exist_file(path_plus_name):
    pass


def remove_folder(old_path, new_path):
    pass


def copy_folder(old_path, new_path):
    pass


def remove_file(old_path_plus_name, new_path_plus_name):
    pass


def copy_file(old_path_plus_name, new_path_plus_name):
    pass


def create_folder(path):
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


def create_file(path_plus_name):
    """
    Check whether a path exists, if not, then create this path.
    Usually embedded in writing operation.
    :param path: i.e., './logs/log.txt' or './logs/'
    :return: create --> True
    """
    flag = False
    if not os.path.exists(os.path.dirname(path_plus_name)):
        try:
            os.makedirs(os.path.dirname(path_plus_name))
            flag = True
        except OSError:
            pass
    return flag


def delete_folder(path):
    pass


def delete_file(path_plus_name):
    pass
