# coding: utf-8
import cv2
import os
import shutil
import math
import numpy as np


folder = 'data/'
show_resize = 3
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

def bubble_sort(rect_list):
    '''
    冒泡排序，稳定的排序，对列表进行排序，从上到下，从左至右
    :param rect_list:
    :return:
    '''
    max_height = 0
    for rect in rect_list:
        temp_width = rect.endy - rect.starty
        if temp_width > max_height:
            max_height = temp_width

    for i in range(0, len(rect_list) - 1):
        for j in range(0, len(rect_list) - 1 - i):
            mid_iy = rect_list[j].starty + (rect_list[j].endy - rect_list[j].starty) / 2
            if mid_iy > rect_list[j + 1].endy:
                # rect_i在rect_j下面，交换
                rect_list[j], rect_list[j + 1] = rect_list[j + 1], rect_list[j]
            elif mid_iy <= rect_list[j + 1].endy and mid_iy >= rect_list[j + 1].starty:
                # rect_i和rect_j在同一行
                if rect_list[j].startx >= rect_list[j + 1].startx:
                    rect_list[j], rect_list[j + 1] = rect_list[j + 1], rect_list[j]
            else:
                # rect_i在rect_j上面
                pass


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
    result2 = cv2.cvtColor(result2, cv2.COLOR_GRAY2BGR)             #颜色转换 RGB转换为灰度图
    # cv2.imwrite(folder + 'temp/thre.jpg', result2)
    i = 0
    for rect in rect_list:
        i = i + 1
        #添加的文字
        # cv2.putText(result2, str(i), (rect.startx - 20, rect.endy + 20),
        #             cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255),
        #             thickness=1,
        #             lineType=1)
        cv2.rectangle(result2, (rect.startx, rect.starty), (rect.endx, rect.endy), (0, 0, 255), 2)
        #画候选框

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
    #cv2.waitKey(1000)                                                  #使窗口始终保持

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

def combine_connected_domain(rect_list):
    '''
    将相邻的连通域融合在一起
    :param rect_list: 连通域矩阵列表
    :return:
    '''
    #if isinstance(rect_list, filter):
    rect_list = list(rect_list)
    # 计算所有框中的高度最大值
    max_height = 0
    for rect in rect_list:
        temp_width = rect.endy - rect.starty
        if temp_width > max_height:
            max_height = temp_width
    # y方向的允许误差设定为最大高度的0.3
    wucha_y = max_height * 0.3

    for i in range(0, len(rect_list) - 1):
        previous = i
        last = i + 1
        while rect_list[previous].startx == -10:
            # 当前矩形已经被合并到前一个,取出前一个进行比较
            previous = previous - 1

        # 两个举行的starty和endy之差在一定范围内
        mid_p = (rect_list[previous].starty + rect_list[previous].endy) / 2
        mid_l = (rect_list[last].starty + rect_list[last].endy) / 2

        if abs(mid_p - mid_l) < wucha_y and rect_list[last].startx - rect_list[previous].endx < max_height:
            rect_list[previous].startx = min(rect_list[previous].startx, rect_list[last].startx)
            rect_list[previous].starty = min(rect_list[previous].starty, rect_list[last].starty)
            rect_list[previous].endx = max(rect_list[previous].endx, rect_list[last].endx)
            rect_list[previous].endy = max(rect_list[previous].endy, rect_list[last].endy)
            rect_list[last].startx = -10
            rect_list[last].starty = -10
            rect_list[last].endx = -10
            rect_list[last].endy = -10
    rect_list = [rect for rect in rect_list if rect.startx != -10]
    return rect_list


def get_single_rect(filename, rect_list):
    '''
    获取单个字符的矩形位置
    :param filename:文件名
    :param list:连通域矩形框列表
    :return:
    '''
    result = thre(filename=filename, is_adapteive=True, thre_value=70)

    list1 = []
    #直接在原图的基础上进行分割，因为有文本框的图，有框作为干扰条件
    #用检测框作为迭代条件

    for rect in rect_list:
        start = 0
        is_start = False
        is_end = False
        is_quanbai = False
        for i in range(rect.startx, rect.endx):
            for j in range(rect.starty, rect.endy):
                if result[j][i] == 0:
                    is_quanbai = False
                    if is_start == False:
                        # 遇到一个黑点，到达起点，保存开始x坐标
                        start = i
                        is_start = True
                    break
                else:
                    is_quanbai = True
            # 到达全白竖线并且已经记录了起点，就设置当前字扫描结束
            if is_quanbai and is_start:
                is_end = True
            # 起点终点扫描结束，保存矩阵信息
            if is_start and is_end:
                if i - start > 7:
                    i = i + 1
                    rect1 = Rect()
                    rect1.startx = start
                    rect1.starty = rect.starty
                    rect1.endy = rect.endy
                    rect1.endx = i
                    list1.append(rect1)
                    is_start = False
                    is_end = False
                    is_quanbai = False
                    start = i
    del rect_list
    return list1

def get_single_rect_without_baibian(filename, list):
    '''
    去除框框里上下的白边
    :param filename:文件名
    :param list: 单个字符框框矩形列表
    :return:
    '''
    result = thre(filename=filename, is_adapteive=True, thre_value=70)
    #去掉白边的方法就是在二维商进行单字检测并分割
    list1 = []
    for rect in list:
        start = 0
        end = 0
        is_start = False
        is_end = False

        for i in range(rect.starty, rect.endy):
            for j in range(rect.startx, rect.endx):
                if result[i][j] == 0:
                    # 遇到一个轮廓框的黑点，到达起点，保存开始x坐标
                    start = i
                    is_start = True
                    break
            if is_start:
                break
        for i1 in range(rect.endy - 1, rect.starty - 1, -1):
            for j1 in range(rect.startx, rect.endx):
                if result[i1][j1] == 0:
                    # 遇到一个黑点，到达起点，保存开始x坐标
                    end = i1
                    is_end = True
                    break
            if is_end:
                break
        # 起点终点扫描结束，保存矩阵信息
        if is_start and is_end:
            rect1 = Rect()
            rect1.startx = rect.startx
            rect1.endx = rect.endx
            rect1.starty = start
            rect1.endy = end

            list1.append(rect1)
    del list
    return list1

def combine_single_bbox(rect_list, jianju=4):
    avg_width = get_avg_height_or_width(rect_list,h_or_w='w') + 10
    print('avg_width:',avg_width)

    # 融合
    for i in range(0, len(rect_list) - 1):
        previous = i
        last = i + 1
        if rect_list[last].starty > rect_list[previous].endy:
            # 换行了
            continue

        if rect_list[previous].endy - rect_list[previous].starty < 30:
            continue
        while rect_list[previous].startx == 0:
            # 当前矩形已经被合并到前一个,取出前一个进行比较
            previous = previous - 1
        if rect_list[last].startx - rect_list[previous].endx < jianju and rect_list[last].endx - rect_list[
            previous].startx < avg_width:
            if rect_list[previous].startx > rect_list[last].startx:
                rect_list[previous].startx = rect_list[last].startx
            if rect_list[previous].starty > rect_list[last].starty:
                rect_list[previous].starty = rect_list[last].starty

            rect_list[previous].startx = min(rect_list[previous].startx, rect_list[last].startx)
            rect_list[previous].endx = max(rect_list[previous].endx, rect_list[last].endx)
            rect_list[previous].starty = min(rect_list[previous].starty, rect_list[last].starty)
            rect_list[previous].endy = max(rect_list[previous].endy, rect_list[last].endy)
            rect_list[last].startx = 0
            rect_list[last].starty = 0
            rect_list[last].endx = 0
            rect_list[last].endy = 0
    rect_list = [rect for rect in rect_list if rect.startx != 0]
    return rect_list


if __name__ == '__main__':
    filename = folder + '4.jpg'
    # 获取连通域
    rect_list = find_contours(filename)
    #通过轮廓检测，给出候选文本框
    draw_rect_line(filename=filename, rect_list=rect_list, windowsname='find_contours', save=False,
                   save_filename='1_find_contours')
    # 连通域排序，从左到右，从上到下
    bubble_sort(rect_list)
    draw_rect_line(filename, rect_list, 'sort', False, 'ocr_yi_sort')
    # 连通域融合
    rect_list = combine_connected_domain(rect_list)
    draw_rect_line(filename, rect_list, 'combine', False, 'ocr_yi_combine')
    #检测单个字符框
    rect_list = get_single_rect(filename, rect_list)
    draw_rect_line(filename, rect_list, 'single', False, 'ocr_yi_single')
    # 去掉上下白边的单字符框
    rect_list = get_single_rect_without_baibian(filename, rect_list)
    draw_rect_line(filename, rect_list, 'get_single_rect_without_baibian', False, 'get_single_rect_without_baibian')
    # 进行融合
    rect_list = combine_single_bbox(rect_list)
    draw_rect_line(filename, rect_list, 'combine_single_bbox', False, 'combine_single_bbox')
    # save_image(filename, rect_list)
    print('finish')
    cv2.waitKey(0)