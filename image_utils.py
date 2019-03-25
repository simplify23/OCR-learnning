# coding: utf-8
import cv2
import os
import shutil
import math
import numpy as np


folder = 'data/'
show_resize = 6
print(os.getcwd())              #返回当前工作目录



class Rect:
    '''
    矩形类
    '''
    startx = 0
    endx = 0
    starty = 0
    endy = 0

    def __init__(self):
        pass

    def __str__(self):
        return '%d %d %d %d' % (self.startx, self.starty, self.endx, self.endy)


def find_contours(filename):
    '''
    对图像进行连通域检测，找到图像中的文字区域
    :param filename:图像名
    :return:包含连通域矩形框的列表
    '''

    # 获取包含矩阵信息的list
    result = thre(filename=filename, is_adapteive=False, thre_value=70)
    #img_show(result, window_name='thre', resize=show_resize)

    cv2.imwrite(folder + 'temp/thre.jpg', result)

    # cv2.bitwise_not(result, dst)                                  #位运算，非
    # 膨胀
    dst = cv2.Canny(result, 100, 100, 3)                            #连通域检测：输入图 maxval minval 卷积大小（默认为3）
    #img_show(dst, window_name='dst', resize=show_resize)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (25, 20))  #获取十字结构元素25*20
    dilated = cv2.dilate(dst, element)
    #img_show(dilated, window_name='dilated', resize=show_resize)


    # 轮廓检测: 输入图 轮廓检测办法（外边缘检测） 近似办法
    #contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #opencv3
    image, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    rect_list = []
    for i in contours:

        first = i[0]
        rect = Rect()
        rect.startx = first[0][0]
        rect.endx = first[0][0]
        rect.starty = first[0][1]
        rect.endy = first[0][1]
        for x in i:
            rect.startx = min(x[0][0], rect.startx)
            rect.endx = max(x[0][0], rect.endx)
            rect.starty = min(x[0][1], rect.starty)
            rect.endy = max(x[0][1], rect.endy)
        rect_list.append(rect)
        '''
        #动态打印轮廓
        img = cv2.drawContours(result, i, -1, (0, 0, 255), 10)
        img_show(img,  window_name='contours',  resize=show_resize)        
        '''
    #打印全部轮廓
    img = cv2.drawContours(result, contours, -1, (0, 255, 0), 10)
    img_show(img, window_name='contours', resize=show_resize)

    avg_height = get_avg_height_or_width(rect_list)
    print('avg_height',avg_height)
    temp_list = [rect for rect in rect_list if
                 rect.endy - rect.starty <= avg_height * 2 and rect.endy - rect.starty > avg_height * 0.7]
    
    return temp_list

def draw_rect_line(filename, rect_list, windowsname="矩形", save=False, save_filename='temp'):
    '''
    在图片上绘制连通域并保存会之后的图片
    :param filename:带绘制图像的的名字
    :param rect_list:连通域矩形列表
    :param windowsname:显示窗口的名字
    :param save:是否保存绘制后的图像
    :param save_filename:保存的图像名字
    :return:None
    '''
    # src = cv2.imread(filename, 0)
    result2 = thre(filename=filename, is_adapteive=True)
    result2 = cv2.cvtColor(result2, cv2.COLOR_GRAY2BGR)
    # cv2.imwrite(folder + 'temp/thre.jpg', result2)
    i = 0
    for rect in rect_list:
        i = i + 1
        # cv2.putText(result2, str(i), (rect.startx - 20, rect.endy + 20),
        #             cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255),
        #             thickness=1,
        #             lineType=1)
        cv2.rectangle(result2, (rect.startx, rect.starty), (rect.endx, rect.endy), (0, 0, 255), 2)

    img_show(result2, windowsname)
    if save:
        if not os.path.exists(folder + 'temp/'):
            os.makedirs(folder + 'temp/')
        cv2.imwrite(folder + 'temp/' + save_filename + '.jpg', result2)

def thre(filename, is_adapteive=False, thre_value=95):
    '''
    对图像进行二值化
    :param filename:图像文件名
    :param is_adapteive: 是否自适应
    :param thre_value: 非自适应模式下的阈值
    :return: 二值化后的图像
    '''
    src = cv2.imread(filename, 0)
    if is_adapteive:
        # result = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 195, 50)
        #输入图 超小阈值赋的值 值域的操作方法 二值化操作类型 分块大小 常数项
        ret, result = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   
        #输入图 阈值 超过或小于阈值赋的值 选择类型 otsu可自动寻找合适阈值
        
    else:
        retval, result = cv2.threshold(src, thre_value, 255, cv2.THRESH_BINARY)

    return result

def img_show(src, window_name='defoult', resize=show_resize):
    '''
    显示图片
    :param src:带显示的图像
    :param window_name:图像窗口名字
    :param resize:resize比例
    :return:None
    '''
    res = cv2.resize(src, (int(src.shape[1] // resize), int(src.shape[0] // resize)), interpolation=cv2.INTER_CUBIC)
    cv2.imshow(window_name, res)
    cv2.waitKey(900)                                                  #使窗口始终保持

def get_avg_height_or_width(rect_list, h_or_w='h'):
    '''
    计算高度的分布，并且得出分布最大区域的平均值
    :param rect_list:连通域矩形列表
    :return:所有连通域中最大的高度
    '''
    # 计算最大值
    max_height = 0
    for rect in rect_list:
        if h_or_w == 'h':
            temp_max_height = rect.endy - rect.starty
        else:
            temp_max_height = rect.endx - rect.startx
        max_height = max((max_height,temp_max_height))

    # 向上取整
    max_height = math.ceil(max_height)

    # 计算分布区间
    if max_height % 10 >= 0:
        max_height = int(max_height // 10 + 1)
    else:
        max_height = int(max_height // 10)
    # 存放每个区间的宽度
    list2 = [[] for i in range(max_height)]
    # 将宽度放入每一个区间
    for rect in rect_list:
        if h_or_w == 'h':
            height = rect.endy - rect.starty
        else:
            height = rect.endx - rect.startx
        if h_or_w=='h':
            if height < 10:
                continue
        temp = height // 10
        list2[int(temp)].append(height)
    # 先取出数量最多的宽度区间
    index_top1 = get_max_height_or_width_index(list2)
    sum_height = sum(list2[index_top1])
    length = len(list2[index_top1])
    del (list2[index_top1])        #删除内部变量对该数据的引用

    # 取出数量最第二多的宽度区间
    index_top1 = get_max_height_or_width_index(list2)

    sum_height += sum(list2[index_top1])
    length += len(list2[index_top1])
    return math.ceil(sum_height / length)


def get_max_height_or_width_index(list2):
    '''
    得到数量最多的高度区间索引
    :param list2: 存储了所有矩形高度信息的列表
    :return:数量最多的高度区间索引
    '''
    max_height = 0
    for i in range(len(list2)-1):
        if len(list2[i]) > max_height:
            max_height = len(list2[i])
            index_top1 = i
    return index_top1

if __name__ == '__main__':
    filename = folder + '4.jpg'
    # 获取连通域
    rect_list = find_contours(filename)
    #通过轮廓检测，给出候选文本框
    draw_rect_line(filename=filename, rect_list=rect_list, windowsname='find_contours', save=True,
                   save_filename='1_find_contours')