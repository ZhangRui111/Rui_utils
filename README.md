# Rui's Utils

There are some useful and common-used utils during my coding issues. So I collect and realize them in Python.

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

### Numpy related

`category/numpy_related.py`

- **save_array_numpy()**:

    Save an numpy array to a text file.

- **read_array_numpy()**:

    Load data from a text file.

- **one_hot_encoding_numpy()**:

    One Hot Encoding using numpy.

- **most_frequent_element_numpy()**: 

    find the most frequent element in a numpy array.

- **normalization_zero2one_numpy()**:

    normalization a numpy array to [0, 1].

### matplotlib.pyplot

`category/matplotlib_pyplot.py`

- **read_output_plot()**: 

    plot numpy array in segment mean value.
    
### folder and file operations

`category/folder_file.py`

- **exist_or_create_path()**:

    Check whether a path exists, if not, then create this path.

def exist_folder(path):

def exist_file(path_plus_name):

def remove_folder(old_path, new_path):

def copy_folder(old_path, new_path):

def remove_file(old_path_plus_name, new_path_plus_name):

def copy_file(old_path_plus_name, new_path_plus_name):

def create_folder(path):

def create_file(path_plus_name):

def delete_folder(path):

def delete_file(path_plus_name):

### multi task

#### 1. multiprocessing

`category/multi_task/multi_processes.py`


- **process_info()**: 

    get (parent/child) process id.

- **multi_process_join()**: 

    multi processes with join to assure order.

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

    info: threading's functions
    
- **thread_object_func()**: 

    info: One thread's function

- **multi_threads_join()**: 

    multi threads with join to assure order.

- **shared_space()**: 

    Using Lock to assure mutually exclusive access.
    
- **multi_threads_queue()**: 

    Use Queue to store multi-threads result.

### More

- **time recoder**: `main.py/main()`

- **Receive args from command line or bat/cmd files.**: `main.py/entry & category.main.bat`
