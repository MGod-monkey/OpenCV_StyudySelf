import cv2 as cv
import numpy as np
import random

def use_capture_video():
    capture = cv.VideoCapture(0)  # 启用摄像头
    while(True):  # 不停读取摄像头捕捉到的图像并形成视屏
        ret,frame = capture.read()
        frame = cv.flip(frame,1)  # 镜像翻转
        cv.imshow('video',frame)
        c = cv.waitKey(1)
        if c == 27:
            break


def get_img_info(img):
    print(img.shape)
    print(img.size)
    print(src.dtype)
    print(np.array(img))    # 将图像转换为一个个二位数组


src = cv.imread('./img/saonv.jpg')  # 载入图片资源
cv.namedWindow('show_img')  # 给窗口命名
cv.imshow('show_img',src)   # 将图像展示在已命名的窗口上
cv.waitKey(0)   # 等待左键按下执行下一步，没有这一步图片将会闪退

use_capture_video()
cv.destroyWindow('show_img')    # 关闭窗口

