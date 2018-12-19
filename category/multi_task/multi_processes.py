"""
Multi threads can only use one core of CPU at a time -- Global Interpreter Lock (GIL).
Multi processes can make fully use of multi-core CPU.

Multi-processes could not have Return value, either (except mp.Pool).
"""
import multiprocessing as mp
import os
import time


def process_info():
    """
    get (parent/child) process id.
    :return:
    """
    print('module name:', __name__)
    print('parent process:', os.getppid())  # get parent process ID.
    print('process id:', os.getpid())  # get current process ID.


def process_job_square(x):
    """
    Function in Pool can have Return value.
    Function in Process can't have Return value.
    :param x:
    :return:
    """
    return x * x


def multicore_pool():
    """
    Pool distribute different process to different CPU core.
    :return:
    """
    # pool = mp.Pool(processes=3) # Define CPU number = 3 (By default, CPU number is all you have.)
    pool = mp.Pool()
    res_0 = pool.map(process_job_square, range(10))
    print(res_0)  # res_0 is the result we get.
    # # apply_async can only handle one iterable args, so comma is necessary.
    res_1 = pool.apply_async(process_job_square, (2,))
    print(res_1.get())
    # # How we handle more than one args with apply_async()
    multi_res = [pool.apply_async(process_job_square, (i,)) for i in range(10)]
    print([res.get() for res in multi_res])


def process_job_sum(l, q):
    """
    Function in Process can't have Return value.
    Use multiprocessing.Queue to store multi-process result.
    :param l: list
    :param q: mp.Queue
    :return:
    """
    res = sum(l)
    q.put(res)


def multi_processes_queue():
    """
    multi-process multiprocessing.Queue to store result.
    :return:
    """
    q = mp.Queue()
    l = []
    for i in range(100):
        l.append(i)
    process_0 = mp.Process(target=process_job_sum, args=(l, q))
    process_1 = mp.Process(target=process_job_sum, args=(l, q))
    process_0.start()
    process_1.start()
    process_0.join()
    process_1.join()
    result_0 = q.get()
    result_1 = q.get()
    print(result_0, ' | ', result_1)
    # multiprocessing.Queue, process or thread can all be placed in it.
    # Queue.qsize()
    # Queue.empty()
    # Queue.full()
    # Queue.get()
    # Queue.get_nowait()
    # Queue.put()
    # Queue.put_nowait()


def process_job_lock(v, num, lock):
    """
    demo process job with multiprocessing lock.
    :param v:
    :param num:
    :param lock:
    :return:
    """
    lock.acquire()
    for _ in range(5):
        time.sleep(0.1)
        v.value += num  # Access shared memory.
        print(v.value)
    lock.release()


def multicore_lock():
    """
    Using Lock to assure mutually exclusive access.
    :return:
    """
    lock = mp.Lock()
    v = mp.Value('i', 0)  # define shared memory, see the appendix
    p1 = mp.Process(target=process_job_lock, args=(v, 1, lock))
    p2 = mp.Process(target=process_job_lock, args=(v, 3, lock))
    p1.start()
    p2.start()
    p1.join()
    p2.join()


def process_job(token):
    """
    demo process job: sleep and print.
    :param token: args
    :return:
    """
    for i in range(20):
        time.sleep(0.1)
        print('process ID: {0} | count: {1} | token: {2}'.format(os.getpid(), i, token))
    print('process ID: {0} is over'.format(os.getpid()))


def multi_process_join():
    """
    multi processes with join to assure order.
    :return:
    """
    process_0 = mp.Process(target=process_job, args=(0,))
    process_1 = mp.Process(target=process_job, args=(1,))
    process_2 = mp.Process(target=process_job, args=(2,))
    # # process_1.daemon == False (default) means child-process can alive even parent process ends.
    # # process_1.daemon == True            means when your script ends its job will kill all subprocess.
    # # it must be set before p.start.
    print(process_1.daemon)
    # process_1.daemon = True
    process_0.start()
    process_0.join()
    process_1.start()
    process_2.start()
    time.sleep(1)
    if process_2.is_alive:
        # stop a process gracefully
        process_2.terminate()
        print('stop process')
        process_2.join()  # Must add *.join() after *.terminate() to update child-process state.


def main():
    # multi_process_join()
    # multicore_lock()
    # multi_processes_queue()
    multicore_pool()


if __name__ == '__main__':
    main()


# ********** Shared Memory ********** #
#
# # 1. shared Value
# value1 = mp.Value('i', 0)
# value2 = mp.Value('d', 3.14)
#
# # 2. shared Array
# array = mp.Array('i', [1, 2, 3, 4])
# # mp.Array() must be one-dimensional array.
# array = mp.Array('i', [[1, 2], [3, 4]]) # This is wrong!
# Error info: TypeError: an integer is required
#
# # 3.All shared Value symbol
# | Type code | C Type             | Python Type       | Minimum size in bytes |
# | --------- | ------------------ | ----------------- | --------------------- |
# | `'b'`     | signed char        | int               | 1                     |
# | `'B'`     | unsigned char      | int               | 1                     |
# | `'u'`     | Py_UNICODE         | Unicode character | 2                     |
# | `'h'`     | signed short       | int               | 2                     |
# | `'H'`     | unsigned short     | int               | 2                     |
# | `'i'`     | signed int         | int               | 2                     |
# | `'I'`     | unsigned int       | int               | 2                     |
# | `'l'`     | signed long        | int               | 4                     |
# | `'L'`     | unsigned long      | int               | 4                     |
# | `'q'`     | signed long long   | int               | 8                     |
# | `'Q'`     | unsigned long long | int               | 8                     |
# | `'f'`     | float              | float             | 4                     |
# | `'d'`     | double             | float             | 8                     |

# ********** Shared Memory ********** #


# ********** Some common errors: ********** #
# 1.
# print(res.get() for res in multi_res)
# Output info: <generator object multicore_pool.<locals>.<genexpr> at 0x000001867ADD6410>
# Correct: print([res.get() for res in multi_res])
# Then output info: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
# 2.
# res_1 = pool.apply_async(process_job_square, (2))
# Error info: TypeError: process_job_square() argument after * must be an iterable, not int
# Correct: res_1 = pool.apply_async(process_job_square, (2,))
# 3.
# res_1 = pool.apply_async(process_job_square, (2, 3, 4,))
# Error info: TypeError: process_job_square() takes 1 positional argument but 3 were given
# Correct: res_1 = pool.apply_async(process_job_square, (2,))

# # Import problems:
# 1.
# import multiprocessing as mp
# ...
# process_0 = mp.Process(target=process_job, args=(0,))
# Error info: AttributeError: module 'multiprocessing' has no attribute 'Process'
# Correct: Rename root folder name from 'multiprocessing' to any name.

# ********** Some common errors: ********** #
