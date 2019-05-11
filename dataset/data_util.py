# encoding = utf-8
import multiprocessing
#多进程包
import threading
#多线程包
import time

import numpy as np

try:
    import queue
except ImportError:
    import Queue as queue


class GeneratorEnqueuer():
    def __init__(self, generator,
                 use_multiprocessing=False,
                 wait_time=0.05,
                 random_seed=None):
        self.wait_time = wait_time
        self._generator = generator
        self._use_multiprocessing = use_multiprocessing
        self._threads = []
        self._stop_event = None
        self.queue = None
        self.random_seed = random_seed

    def start(self, workers=1, max_queue_size=10):
        '''
        选择是否调用多线程，并通过守护进程，开始同时处理多张图片

        :param workers:  需要多少个工作线程？？
        :param max_queue_size: 最大的队列尺寸，同时处理10张图？？
        :return:
        '''
        def data_generator_task():
            '''
            把三个迭代的信息打包成一个对象并压入多线程队列中，进行同时处理
            :return:
            '''
            while not self._stop_event.is_set():
                #event：使用set()方法后，isSet()方法返回True
                try:
                    if self._use_multiprocessing or self.queue.qsize() < max_queue_size:
                        generator_output = next(self._generator)
                        #generator有三个迭代对象：图片 检测框坐标 图片尺寸信息
                        #将三个作为一个对象入队
                        self.queue.put(generator_output)
                        #print('队列入了{}'.format(generator_output))
                    else:
                        time.sleep(self.wait_time)
                except Exception:
                    self._stop_event.set()
                    # event.set()：将event的标志设置为True，调用wait方法的所有线程将被唤醒。
                    raise
                #当程序出现错误，python会自动引发异常，也可以通过raise显示地引发异常。

        #这才是主函数，天呐
        try:
            #选择是否使用多进程
            if self._use_multiprocessing:
                self.queue = multiprocessing.Queue(maxsize=max_queue_size)
                self._stop_event = multiprocessing.Event()
            else:
                self.queue = queue.Queue()
                self._stop_event = threading.Event()

            for _ in range(workers):
                if self._use_multiprocessing:
                    # Reset random seed else all children processes
                    # share the same seed
                    np.random.seed(self.random_seed)
                    thread = multiprocessing.Process(target=data_generator_task)
                    #process(target): 要执行的方法
                    thread.daemon = True
                    #当且仅当主线程运行时有效，当其他非Daemon线程结束时可自动杀死所有Daemon线程。
                    if self.random_seed is not None:
                        self.random_seed += 1
                else:
                    thread = threading.Thread(target=data_generator_task)
                self._threads.append(thread)
                thread.start()
        except:
            self.stop()
            raise

    def is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()
        #是否正在运行
        #信号传输没有被阻塞并且有事件？

    def stop(self, timeout=None):
        '''
        关闭对应的多线程或者多进程
        :param timeout:
        :return:
        '''
        if self.is_running():
            self._stop_event.set()

        #线程工作情况
        for thread in self._threads:
            if thread.is_alive():
            #thread.isAlive(): 返回线程是否活动的。
                if self._use_multiprocessing:
                    thread.terminate()
                    #TerminateThread在线程外终止一个线程，用于强制终止线程。
                else:
                    thread.join(timeout)
                #thread.join()：(用户进程)所完成的工作就是线程同步，
                #即主线程任务结束之后，进入阻塞状态，一直等待其他的子线程执行结束之后，主线程再终止

        #进程工作状况
        if self._use_multiprocessing:
            if self.queue is not None:
                self.queue.close()

        self._threads = []
        self._stop_event = None
        self.queue = None

    def get(self):
        while self.is_running():
            if not self.queue.empty():
                inputs = self.queue.get()
                if inputs is not None:
                    yield inputs
            else:
                time.sleep(self.wait_time)
