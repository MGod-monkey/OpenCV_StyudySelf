import cv2 as cv
import numpy as np


# 提升图片的对比度和亮度
def update_light(img_path,d,l):
    img = cv.imread(img_path)
    h,w,ch = img.shape
    blank = np.zeros([h,w,ch], img.dtype)   # 创建一个与图像相同规模的数组
    dst = cv.addWeighted(img,d,blank,1-d,l)
    cv.imshow('update_light',dst)
    cv.imshow('img',img)
    cv.imwrite('./img/lena_light.jpg',dst)


if __name__ == '__main__':
    update_light('./img/lena.jpg',1.2,60)
    cv.waitKey(0)
    cv.destroyAllWindows()