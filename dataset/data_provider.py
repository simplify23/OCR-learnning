# encoding:utf-8
'''
读取处理过的标签和图片
'''
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.dataset.data_util import GeneratorEnqueuer

DATA_FOLDER = "../../data/dataset/mlt/"


def get_training_data():
    '''

    :return 找到的训练集img的地址
    '''
    img_files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(os.path.join(DATA_FOLDER, "image")):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    img_files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(img_files)))
    return img_files


def load_annoataion(p):
    '''
    读取文件中的坐标点
    :param p:  文件指针
    :return: 以坐标点为列表的各个框
    '''
    bbox = []
    with open(p, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split(",")
        x_min, y_min, x_max, y_max = map(int, line)
        bbox.append([x_min, y_min, x_max, y_max, 1])
    return bbox


def generator(vis=False):
    '''
    读取处理过的标签和图片
    :param vis: 是否选择展示效果
    :return:
    [im]：图片列表
    Bbox：检测框坐标
    im_info: 图片——长宽通道数——信息
    '''
    image_list = np.array(get_training_data())
    #print(image_list)
    print('{} training images in {}'.format(image_list.shape[0], DATA_FOLDER))
    index = np.arange(0, image_list.shape[0])
    while True:
        np.random.shuffle(index)
        #把顺序打乱
        for i in index:
            try:
                im_fn = image_list[i]
                im = cv2.imread(im_fn)
                h, w, c = im.shape
                im_info = np.array([h, w, c]).reshape([1, 3])
                #读取图片

                #解析出标签
                _, fn = os.path.split(im_fn)
                #路径 文件名
                fn, _ = os.path.splitext(fn)
                #文件名 扩展名
                txt_fn = os.path.join(DATA_FOLDER, "label", fn + '.txt')
                #组合出对应的txt标签文件
                if not os.path.exists(txt_fn):
                    print("Ground truth for image {} not exist!".format(im_fn))
                    continue
                bbox = load_annoataion(txt_fn)
                if len(bbox) == 0:
                    print("Ground truth for image {} empty!".format(im_fn))
                    continue

                #将拼接出来的检测矩形框显示到图片上
                if vis:
                    for p in bbox:
                        cv2.rectangle(im, (p[0], p[1]), (p[2], p[3]), color=(0, 0, 255), thickness=1)
                    fig, axs = plt.subplots(1, 1, figsize=(30, 30))
                    axs.imshow(im[:, :, ::-1])
                    #设置坐标刻度
                    axs.set_xticks([])
                    axs.set_yticks([])
                    plt.tight_layout()
                    #plt.tight_layout会自动调整子图参数，使之填充整个图像区域。
                    plt.show()
                    plt.close()
                    #plt.imshow()函数负责对图像进行处理，并显示其格式，但是不能显示。
                    #其后跟着plt.show()才能显示出来。
                yield [im], bbox, im_info

            except Exception as e:
                print(e)
                continue


def get_batch(num_workers, **kwargs):
    #**kwargs：形参中按照关键字传值把多余的传值以字典的方式呈现
    '''
    在实现按批次处理？？
    将各张图片以三个对象压人队列信息进行处理（内部主要实现了多线程，并没有具体处理？？）
    :param num_workers:
    :param kwargs:
    :return:
    '''
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        #传入的第一个参数为迭代器
        enqueuer.start(max_queue_size=24, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    #put放进去 get拿出来
                    #拿到处理图片的某一进程
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        #finally块的作用就是为了保证无论出现什么情况，finally块里的代码一定会被执行。
        #结束进程
        if enqueuer is not None:
            enqueuer.stop()


if __name__ == '__main__':
    gen = get_batch(num_workers=2, vis=False)
    while True:
        image, bbox, im_info = next(gen)
        #将三个信息按原格式出队列
        print('done')
