'''
2019/5/5
reader: zhengtianlun
fuction:  将图片放大到1200*600大小，将坐标标签分割成固定的16像素矩形框

边缘检测？？
'''
# coding = utf-8
import os
import sys

import cv2 as cv
import numpy as np
from tqdm import tqdm

sys.path.append(os.getcwd())
#其实sys.path是一个列表，这个列表内的路径都添加到环境变量中去了。
#sys.path.append()方法可以添加自定义的路径。
#os.getcwd()： 返回当前目录
from utils import orderConvex, shrink_poly
#shrink：收缩  convex：凸面
print(os.getcwd())
#DATA_FOLDER = "/media/D/DataSet/mlt_selected/"
DATA_FOLDER = "../../mlt_selected/"
#/media：为绝对路径 media为相对路径
OUTPUT = "../../data/dataset/mlt/"
MAX_LEN = 1200
MIN_LEN = 600

im_fns = os.listdir(os.path.join(DATA_FOLDER, "image"))
#os.listdir() 返回指定的文件夹包含的文件或文件夹的名字的列表。
#os.path.join()函数用于路径拼接文件路径。
im_fns.sort()

if not os.path.exists(os.path.join(OUTPUT, "image")):
    os.makedirs(os.path.join(OUTPUT, "image"))
if not os.path.exists(os.path.join(OUTPUT, "label")):
    os.makedirs(os.path.join(OUTPUT, "label"))

for im_fn in tqdm(im_fns):
    try:
        # 处理标签，获得答案
        _, fn = os.path.split(im_fn)
        # os.path.split（）返回文件的路径和文件名
        bfn, ext = os.path.splitext(fn)
        # os.path.splitext()将文件名和扩展名分开
        if ext.lower() not in ['.jpg', '.png']:
            continue

        gt_path = os.path.join(DATA_FOLDER, "label", 'gt_' + bfn + '.txt')
        img_path = os.path.join(DATA_FOLDER, "image", im_fn)

        img = cv.imread(img_path)
        img_size = img.shape
        im_size_min = np.min(img_size[0:2])
        im_size_max = np.max(img_size[0:2])
        #宽 长 RGB（3）这里意思是只取长和宽

        #等比例缩放图片尺寸
        im_scale = float(600) / float(im_size_min)
        if np.round(im_scale * im_size_max) > 1200:
        #np.round 返回浮点数的四舍五入值
            im_scale = float(1200) / float(im_size_max)
        new_h = int(img_size[0] * im_scale)
        new_w = int(img_size[1] * im_scale)

        new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
        new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16
        #向上取整

        re_im = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
        re_size = re_im.shape

        #读取标签并获得标签检测框坐标
        polys = []
        with open(gt_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            splitted_line = line.strip().lower().split(',')
            #str.strip()就是把字符串(str)的头和尾的空格，以及位于头尾的\n \t之类给删掉。
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, splitted_line[:8])
            #map() 会根据提供的函数对指定序列做映射。
            #第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表。
            poly = np.array([x1, y1, x2, y2, x3, y3, x4, y4]).reshape([4, 2])
            #因已对图片缩放，故等比例对图片的检测坐标进行缩放
            poly[:, 0] = poly[:, 0] / img_size[1] * re_size[1]
            poly[:, 1] = poly[:, 1] / img_size[0] * re_size[0]
            poly = orderConvex(poly)
            polys.append(poly)
            #poly 0/1代表x坐标，y坐标

            #cv.polylines(re_im, [poly.astype(np.int32).reshape((-1, 1, 2))], True,color=(0, 255, 0), thickness=2)
            #polylines：绘制多边形  thickness:折线粗细  astype：修改数据类型
            #代码的作用为用大框框出最大文本

        res_polys = []
        for poly in polys:
            # delete polys with width less than 10 pixel
            if np.linalg.norm(poly[0] - poly[1]) < 10 or np.linalg.norm(poly[3] - poly[0]) < 10:
            #np.linalg.norm：线性代数的范数
                continue

            res = shrink_poly(poly)
            #
            #for p in res:
            #    cv.polylines(re_im, [p.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=1)
            #代码的作用为：微分成绿色矩形 BGR
            res = res.reshape([-1, 4, 2])
            for r in res:
                x_min = np.min(r[:, 0])
                y_min = np.min(r[:, 1])
                x_max = np.max(r[:, 0])
                y_max = np.max(r[:, 1])

                res_polys.append([x_min, y_min, x_max, y_max])

        cv.imwrite(os.path.join(OUTPUT, "image", fn), re_im)
        with open(os.path.join(OUTPUT, "label", bfn) + ".txt", "w") as f:
            for p in res_polys:
                line = ",".join(str(p[i]) for i in range(4))
                f.writelines(line + "\r\n")


                #for p in res_polys:
                #    cv.rectangle(re_im,(p[0],p[1]),(p[2],p[3]),color=(0,0,255),thickness=1)
                #代码的作用为：微分成红色矩形 BGR
                #cv.imshow("demo",re_im)
                #cv.waitKey(0)
    except:
        print("Error processing {}".format(im_fn))
