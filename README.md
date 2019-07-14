# OCR-learnning
这是个学习别人零碎项目片段，然后运行的项目。为了学习，本人主要是跑通代码，并添加大量注释。
copy过来不一定能运行，后期整理集成到自己代码

环境配置要求：  
ubuntu环境下:  
opencv v4.0.0.21

1. image util3 ：用opencv检测文本已经可以使用
来源 ：师兄的代码

2. nms.py  | predict.py 
来源 ：advanceEAST 
https://github.com/huoyijie/AdvancedEAST

3.folder prepar/ | dataset/
来源：CTPN（TF版本）
https://github.com/eragonruan/text-detection-ctpn

因为cython编译复杂,很容易出错,所以把原版的ctpn/lib的utils已编译文件上传
