import os
import pathlib
import shutil


def exist_or_create_folder(path_name):
    """
    Check whether a path exists, if not, then create this path.
    :param path_name: i.e., './logs/log.txt' or './logs/'
    :return: flag == False: failed; flag == True: successful.
    """
    flag = False
    pure_path = os.path.dirname(path_name)
    if not os.path.exists(pure_path):
        try:
            os.makedirs(pure_path)
            flag = True
        except OSError:
            pass
    return flag


def exist_folder(path):
    """
    Whether a folder exists.
    :param path: folder path: i.e., './logs/log.txt' or './logs/'
    :return:
    """
    pure_path = os.path.dirname(path)
    return os.path.exists(pure_path)


def exist_file(path_name):
    """
    Whether a file exists.
    :param path_name: file path.
    :return:
    """
    return os.path.isfile(path_name)
    # or return os.path.exists(path_name)


def move_folder(source_path, dest_path):
    """
    Move a folder from source_path to dest_path, including its contents.
    :param source_path: 'test0/test12/'
    :param dest_path: 'test1/test12/'
    :return: flag == False: failed; flag == True: successful.
    """
    flag = False
    pure_source_path = os.path.dirname(source_path)
    pure_dest_path = os.path.dirname(dest_path)
    if os.path.exists(pure_source_path):
        shutil.move(pure_source_path, pure_dest_path)
        flag = True
    else:
        pass
        # raise Exception('move folder error!')
    return flag


def copy_folder(source_path, dest_path):
    """
    Copy a folder from source_path to dest_path, including its contents.
    :param source_path: 'test0/test12/'
    :param dest_path: 'test1/test12/'  cannot exist before.
    :return: flag == False: failed; flag == True: successful.
    """
    flag = False
    pure_source_path = os.path.dirname(source_path)
    pure_dest_path = os.path.dirname(dest_path)
    if os.path.exists(pure_source_path):
        shutil.copytree(pure_source_path, pure_dest_path)
        flag = True
    else:
        print('copy folder error!')
    return flag


def move_file(source_path, dest_path):
    """
    Move a file from source_path to dest_path.
    :param source_path: 'test0/test12/test.txt'
    :param dest_path: 'test1/test12/test.txt' or 'test1/test12/'
    :return: flag == False: failed; flag == True: successful.
    """
    flag = False
    if os.path.exists(source_path):
        shutil.move(source_path, dest_path)
        flag = True
    else:
        print('move file error!')
    return flag


def copy_file(source_path, dest_path):
    """
    Copy a file from source_path to dest_path.
    :param source_path: see annotation.
    :param dest_path: see annotation.
    :return: flag == False: failed; flag == True: successful.
    """
    flag = False
    if os.path.exists(source_path):
        shutil.copyfile(source_path, dest_path)  # source_path, dest_path must be files.
        # shutil.copy(source_path, dest_path)  # source_path must be a folder, dest_path may be a folder/a file.
        flag = True
    else:
        print('copy file error!')
    return flag


def create_folder(path):
    """
    create a folder.
    :param path: i.e., './logs/log.txt' or './logs/'
    :return: flag == False: path doesn't exists; flag == True: delete successfully.
    """
    flag = False
    pure_path = os.path.dirname(path)
    if not os.path.exists(pure_path):
        try:
            os.makedirs(pure_path)
            flag = True
        except OSError:
            pass
    return flag


def delete_folder(path):
    """
    Remove/delete a folder as well as all its content.
    :param path:
    :return: flag == False: path doesn't exists; flag == True: delete successfully.
    """
    flag = False
    pure_path = os.path.dirname(path)
    if os.path.exists(pure_path):
        try:
            shutil.rmtree(pure_path)
            flag = True
        except OSError:
            pass
    return flag


def delete_file(path_name):
    """
    Remove/delete a file.
    :param path_name:
    :return: flag == False: file doesn't exists; flag == True: delete successfully.
    """
    flag = False
    if os.path.isfile(path_name):
        try:
            os.remove(path_name)
            # # or
            # os.unlink(path_name)
            flag = True
        except OSError:
            pass
    return flag


def rename(source_path, dest_path):
    """
    Rename a file or a folder.
    :param source_path: old name
    :param dest_path: new name
    :return: flag == False: file doesn't exists; flag == True: delete successfully.
    """
    flag = False
    if os.path.exists(source_path):
        try:
            os.rename(source_path, dest_path)
            flag = True
        except OSError:
            pass
    return flag


def main():
    parent_path = '../test/'
    print(rename(parent_path + '../test1/', parent_path))


if __name__ == '__main__':
    main()

# ------------------ remove/delete -------------------- #

# os.remove()/os.unlink() removes a file.
# pathlib.Path.unlink() removes the file or symbolic link.
# os.rmdir() removes an empty directory.
# shutil.rmtree() deletes a directory and all its contents.
# pathlib.Path.rmdir() removes the empty directory.

# ------------------ remove/delete -------------------- #
