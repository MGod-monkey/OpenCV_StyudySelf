import cv2 as cv
import numpy as np


# 计时装饰器
def count_time(fn):
    def inner(img=None):
        start = cv.getTickCount()
        fn(img)
        end = cv.getTickCount()
        count = (end-start)/cv.getTickFrequency()
        print('函数[{}]耗时{:.5f} ms'.format(fn.__name__,count*1000))
    return inner


# 调用摄像头模块
def use_capture_video():
    capture = cv.VideoCapture(0)  # 启用摄像头
    while True:  # 不停读取摄像头捕捉到的图像并形成视屏
        ret,frame = capture.read()
        frame = cv.flip(frame,1)  # 镜像翻转
        cv.imshow('video',frame)
        c = cv.waitKey(1)
        if c == 27:
            break


# 获取图像信息
def get_img_info(img):
    print(img.shape)
    print(img.size)
    print(src.dtype)
    print(np.array(img))    # 将图像转换为一个个二位数组


# 对图像进行像素取反
@count_time
def reverse_pixels(img):
    new_img = cv.bitwise_not(img)
    #new2_img = cv.bitwise_or(img,new_img)
    #cv.imshow('saonv_or',new2_img)
   # new3_img = cv.bitwise_and(img,new_img)
    #cv.imshow('saonv_and',new3_img)
    cv.imwrite('./img/saonv_reverse.jpg',new_img)
    cv.imshow('saonv_reverse',new_img)


src = cv.imread('./img/saonv.jpg')  # 载入图片资源
cv.namedWindow('show_img')  # 给窗口命名
cv.imshow('show_img',src)   # 将图像展示在已命名的窗口上
reverse_pixels(src)
use_capture_video()
cv.waitKey(0)
cv.destroyWindow('show_img')    # 关闭窗口

