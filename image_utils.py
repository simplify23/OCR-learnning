import cv2
import os
import shutil
import math
import numpy as np


folder = 'data/'
show_resize = 2
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
    img_show(result, window_name='thre', resize=show_resize)

    cv2.imwrite(folder + 'temp/thre.jpg', result)

    # cv2.bitwise_not(result, dst)                                  #位运算，非
    # 膨胀
    dst = cv2.Canny(result, 100, 100, 3)                            #连通域检测：输入图 maxval minval 卷积大小（默认为3）
    img_show(dst, window_name='dst', resize=show_resize)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (25, 20))  #获取十字结构元素25*20
    dilated = cv2.dilate(dst, element)
    img_show(dilated, window_name='dilated', resize=show_resize)


    # 轮廓检测: 输入图 轮廓检测办法（外边缘检测） 近似办法
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #opencv3
    #image, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


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

    '''avg_height = get_avg_height_or_width(rect_list)
    print('avg_height',avg_height)
    temp_list = [rect for rect in rect_list if
                 rect.endy - rect.starty <= avg_height * 2 and rect.endy - rect.starty > avg_height * 0.7]
    
    return temp_list    '''
    return 0

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
        #输入图 阈值 超过或小于阈值赋的值 选择类型
        
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
    cv2.waitKey(1500)                                                  #使窗口始终保持


if __name__ == '__main__':
    filename = folder + '4.jpg'
    # 获取连通域
    rect_list = find_contours(filename)