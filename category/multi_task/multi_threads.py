import threading
import time

from queue import Queue
# # Multi-threads could not have Return value. Use Queue to store multi-threads result.


def threading_func(index):
    """
    info: threading's functions
    :param index:
    :return:
    """
    if index == 0:
        print(threading.active_count())  # Return the number of Thread objects currently alive.
        return threading.active_count()
    elif index == 1:
        print(threading.enumerate())  # Return a list of all Thread objects currently alive.
        return threading.enumerate()
    elif index == 2:
        print(threading.get_ident())  # Return the ‘thread identifier’ of the current thread.
        return threading.get_ident()
    elif index == 3:
        print('current_thread {}'.format(threading.current_thread()))  # Return the current Thread object.
        return threading.current_thread()
    else:
        # Return the main Thread object, which the Python interpreter was started.
        print('main_thread {}'.format(threading.main_thread()))
        return threading.main_thread()


def thread_object_func(thread_object, index):
    """
    info: One thread's function
    :param thread_object:
    :param index:
    :return:
    """
    if index == 0:
        thread_object.start()  # Start the thread’s activity.
    elif index == 1:
        # A boolean value indicating whether this thread is a daemon thread (True) or not (False).
        print('This thread is a daemon thread {}'.format(thread_object.daemon))
    elif index == 2:
        thread_object.join()  # Wait until the thread terminates, which could be used to assure orders.
    elif index == 3:
        thread_object.is_alive()  # Return whether the thread is alive.
    else:
        thread_object.run()  # Method representing the thread’s activity (Its use is not recommended).


def thread_job_square(l, q):
    """ Store element's square.
    :param l: list
    :param q: queue
    :return:
    """
    for i in range(len(l)):
        l[i] = l[i] ** 2
    q.put(l)
    threading_func(4)
    threading_func(3)
    print('size of queue: {}'.format(q.qsize()))


def multi_threads_queue():
    """
    Use Queue to store multi-threads result.
    :return:
    """
    q = Queue()
    threads = []
    results = []
    data = [[1, 2, 3], [3, 4, 5], [4, 4, 4], [5, 5, 5]]
    for i in range(len(data)):
        t = threading.Thread(target=thread_job_square, args=(data[i], q))
        t.start()
        threads.append(t)
    print('size of queue: {}'.format(q.qsize()))  # ``q.qsize()'' returns 4, meaning all threads have finished.
    for thread in threads:
        thread.join()
    # # *.join() assures that executing ``q.get()'' after finishing every thread's calculation.
    for _ in range(len(data)):
        results.append(q.get())
    print(results)


def shared_space_job1():
    """
    demo thread job with thread lock.
    :return:
    """
    global A, lock
    lock.acquire()
    for i in range(10):
        A += 1
        print('job1', A)
    lock.release()


def shared_space_job2():
    """
    demo thread job with thread lock.
    :return:
    """
    global A, lock
    lock.acquire()
    for i in range(10):
        A += 10
        print('job2', A)
    lock.release()


def shared_space():
    """
    Using Lock to assure mutually exclusive access.
    :return:
    """
    t1 = threading.Thread(target=shared_space_job1())
    t2 = threading.Thread(target=shared_space_job2())
    t1.start()
    t2.start()
    t1.join()
    t2.join()


def thread_job(token):
    """
    demo thread job: Print something.
    :param token: args
    :return:
    """
    for i in range(25):
        time.sleep(0.1)
        print('thread ID: {0} | count: {1} | token: {2}'.format(threading.get_ident(), i, token))
    print('thread ID: {0} is over'.format(threading.get_ident()))


def multi_threads_join():
    """
    multi threads with join to assure order.
    :return:
    """
    # thread_0 = threading.Thread(group=None, target=thread_job, name=None, args=(), daemon=None)
    thread_0 = threading.Thread(target=thread_job, args=(0,), daemon=False)
    thread_1 = threading.Thread(target=thread_job, args=(1,), daemon=False)
    thread_2 = threading.Thread(target=thread_job, args=(2,), daemon=False)
    thread_0.start()
    thread_0.join()
    thread_1.start()
    thread_2.start()


def main():
    # multi_threads_join()
    # shared_space()
    multi_threads_queue()


if __name__ == '__main__':
    # Global variables must be here.
    lock = threading.Lock()
    A = 0

    main()


# # Some common errors:
# 1.
# thread_0 = threading.Thread(target=thread_job, args=(0), daemon=False)
# Error info: TypeError: thread_job() argument after * must be an iterable, not int
# Correct: thread_0 = threading.Thread(target=thread_job, args=(, ), daemon=False)
# 2.
# t = threading.Thread(target=thread_job_square(), args=(data[i], q))
# Error info: TypeError: thread_job_square() missing 2 required positional arguments: 'l' and 'q'
# Correct: t = threading.Thread(target=thread_job_square, args=(data[i], q))

# # Some tips:
# 1.
# Thread's running starts from ``thread_object.start()''. Or we can see ``*.start()'' as a declaration
# that this thread is ready, CPU can running it (Usually it will be run instantly).
