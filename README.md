# Rui's Utils

There are some useful and common-used utils during my coding issues. So I collect and realize them in Python. 
Note: **all codes are compatible with python 3.X**.

## Content

### Save and Read

`category/save_and_read.py`

- **save_txt()**:

    Save to a file.

- **read_txt()**:

    Read from a file.

- **save_csv()**:

    Save to CSV file.

- **read_csv()**: 

    Read CSV file.

- **save_array_numpy()**: 

    Save an numpy array to a text file.

- **read_array_numpy()**:

    Load data from a text file.

- **save_dic_txt()**: 

    Save a dictionary to a .txt file.

- **read_dic_txt()**:

    Read a dictionary from a .txt file.

### Numpy related

`category/numpy_related.py`

- **save_array_numpy()**:

    Save an numpy array to a text file.

- **read_array_numpy()**:

    Load data from a text file.

- **one_hot_encoding_numpy()**:

    One Hot Encoding using numpy.

- **most_frequent_element_numpy()**: 

    Find the most frequent element in a numpy array.

- **normalization_zero2one_numpy()**:

    Normalization a numpy array to [0, 1].
    
- **conditional_indexes()**: 

    np.where to get conditional indexes.

- **merge_array_lists2single()**:

    Convert list of numpy arrays (or list of list) into single numpy array.

- **count_items()**: 

    Count the occurrence of certain item in an ndarray.
### List related

`category/list_related.py`

- **remove_duplicates()**:

    Remove Duplicates From a Python List.
    
- **convert_items_type()**:

    Convert all items in a list to int/float/str type.

### matplotlib.pyplot

`category/matplotlib_pyplot.py`

- **read_output_plot()**:

    Plot numpy array in segment mean value.

- **show_gray_image()**:

    Show a numpy array ao a gray image.

- **aspect_ratio_image()**:

    Change the aspect ratio of multiple image (or axes).

- **aspect_ratio_multi_image()**:

    Change the aspect ratio of multiple image (or axes)
    so that they have exactly the same display aspect ratio.

### folder and file operations

`category/folder_file.py`

- **exist_or_create_folder(path_name)**:

    Check whether a path exists, if not, then create this path.

- **exist_folder(path)**:

    Whether a folder exists.

- **exist_file(path_name)**:

    Whether a file exists.

- **move_folder(source_path, dest_path)**:

    Move a folder from source_path to dest_path, including its contents.

- **copy_folder(source_path, dest_path)**:

    Copy a folder from source_path to dest_path, including its contents.

- **move_file(source_path, dest_path)**:

    Move a file from source_path to dest_path.

- **copy_file(source_path, dest_path)**:

    Copy a file from source_path to dest_path.

- **create_folder(path)**:

    Create a folder.

- **delete_folder(path)**:

    Remove/delete a folder as well as all its content.

- **delete_file(path_name)**:

    Remove/delete a file.

- **rename(source_path, dest_path)**:

    Rename a file or a folder.


### multi task

#### 1. multiprocessing

`category/multi_task/multi_processes.py`


- **process_info()**: 

    Get (parent/child) process id.

- **multi_process_join()**: 

    Multi processes with join to assure order.

- **multicore_lock()**: 

    Using Lock to assure mutually exclusive access.

- **multi_processes_queue()**: 

    Function in Process can't have Return value.
    Use multiprocessing.Queue to store multi-process result.

- **multicore_pool()**: 

    Function in Pool can have Return value.
    Pool distribute different process to different CPU core.
    
#### 2. threading

`category/multi_task/multi_threads.py`

- **threading_func()**: 

    Info: threading's functions
    
- **thread_object_func()**: 

    Info: One thread's function

- **multi_threads_join()**: 

    Multi threads with join to assure order.

- **shared_space()**: 

    Using Lock to assure mutually exclusive access.
    
- **multi_threads_queue()**: 

    Use Queue to store multi-threads result.

### Tensorflow related

`category/tensorflow_related.py`

- **whether_differentiable()**: 

    Use tf.gradient to find out whether an operation is derivable in Tensorflow (< 1.8.0).
    If there are some operations nondifferentiable in y, error appears.

- **main() + tf.app.run() + tf.flags**

    Receive parameters from terminal.
    
### Torch related

`category/ddp.py`: Data Parallel in PyTorch by torch.nn.parallel.DistributedDataParallel

`category/torch_related.py`

- **Allocate program to a special GPU**:

    `category/torch_related.py/main()` 

- **index_by_array()**: 

    Array as index in torch.
    
- **torch_logical_operations()**:

    Torch logical operations: or, and, not, xor. 
    
- **get_index_by_value()**:

    Get index by value.

- **one_hot_torch()**:

    One-hot transformation in torch.

### More

`category/more.py`

- **binary_array_to_int()**: 

    Transfer an bin-array to int(scalar).
    
- **variable_exist_before()**: 

    To check whether a variable exists before now.
    
- **argparse**

    Receive parameters from terminal. `main()`

### Machine Learning

`category/ML/*`

- **PCA_sklearn.py**: 

    Apply PCA to the dataset using sklearn.
    
- **shuffle_data.py**: 

    Shuffle the data (features and labels) using sklearn or numpy.
    
- **load_datasets.py**: 

    Load mnist, cifar10 or iris datasets.
    
- **get_dataset_stats.py**

    Get statistic information (i.e., mean and std) of some typical dataset.


### Others

- **Time recoder**: `main.py/main()`
- **Receive parameters from terminal or bat/cmd files by sys.argv.**: `main.py/entry & category/main.bat`
- **Delete files under a folder and bypass "Are you sure you want to delete..." prompt**: `category/delete_files.bat`
- **Catch KeyboardInterrupt [ctrl + C]**: `main.py/line 95`
- **Add new directories to the path attribute of the sys module**: `main.py/line 5 & 12`
- **Check if an object has an attribute in Python**: `main.py/line 81`
