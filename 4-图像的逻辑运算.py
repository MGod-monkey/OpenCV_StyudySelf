import cv2 as cv
import numpy as np


def lj_operation(img1_path,img2_path):
    img1 = cv.imread(img1_path)
    img2 = cv.imread(img2_path)
    img1_and_img2 = cv.add(img1,img2)   # 相加运算
    cv.imwrite('img/素材/img1_and_img2.jpg', img1_and_img2)
    cv.imshow('img_and',img1_and_img2)
    img1_sub_img2 = cv.subtract(img1,img2)  # 相减运算
    cv.imwrite('img/素材/img1_sub_img2.jpg', img1_sub_img2)
    cv.imshow('img_sub', img1_sub_img2)
    img2_sub_img1 = cv.subtract(img2, img1)  # 相减运算
    cv.imwrite('img/素材/img2_sub_img1.jpg', img2_sub_img1)
    cv.imshow('img2_sub', img2_sub_img1)


if __name__ == '__main__':
    lj_operation('img/素材/LinuxLogo.jpg', 'img/素材/WindowsLogo.jpg')
    cv.waitKey(0)
    cv.destroyAllWindows()