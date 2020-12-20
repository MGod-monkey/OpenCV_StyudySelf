import cv2 as cv
import numpy as np


# 创建一个纯色图片(RGB通道)
def create_img(RGB):
    img = np.zeros([450,450,3],np.uint8)
    img.fill()
    img[:,:,0] = np.ones([450,450]) * RGB[0]   # B通道的值
    img[:,:,1] = np.ones([450,450]) * RGB[1]    # G通道的值
    img[:,:,2] = np.ones([450,450]) * RGB[2]    # R通道的值
    cv.imshow('show_RGB',img)


# 创建一个灰度纯色图片（单通道）
def create_grey_img(grey_degree):
    img = np.ones([250,250,1],np.uint8) * grey_degree
    cv.imshow('show_grey_img',img)

"""
src = cv.imread('./img/saonv.jpg')  # 载入图片资源
create_img((185,206,117))
create_grey_img(124)    # 灰度图像从0~255依次变亮，即从黑到白
cv.waitKey(0)   # 等待左键按下执行下一步，没有这一步图片将会闪退
cv.destroyAllWindows()    # 关闭窗口
"""
array = np.array([[1,2,3],[4,5,6],[7,8,9]],np.uint8)
print(array)
array1 = array.reshape([1,9,1])
print(array1)



