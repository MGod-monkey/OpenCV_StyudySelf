import cv2 as cv
import numpy as np


# 图像色彩分离(cv2.inRange())
def img_color_depart(img_path,img2_path):
    img = cv.imread(img_path)
    img2 = cv.imread(img2_path)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # 将图像转换为hsv形式
    img2_hsv = cv.cvtColor(img2, cv.COLOR_BGR2HSV)  # 将图像转换为hsv形式
    lower_white_hsv = np.array([0, 0, 221])  # hsv中的白色过滤取值范围
    upper_white_hsv = np.array([180, 30, 255])
    lower_black_hsv = np.array([0, 0, 0])  # hsv中的黑色过滤取值范围
    upper_black_hsv = np.array([180, 255, 46])
    mask = cv.inRange(img_hsv, lower_white_hsv, upper_white_hsv)
    mask2 = cv.inRange(img2_hsv, lower_black_hsv, upper_black_hsv)
    cv.imwrite('img/素材/saonv_mask.jpg', mask)
    cv.imshow('img_mask', mask)
    cv.imwrite('img/素材/saonv_reverse_mask.jpg', mask2)
    cv.imshow('img_reverse_mask', mask2)


# 视频色彩分离(cv2.inRange())
def video_color_depart():
    capture = cv.VideoCapture(0)
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv.flip(frame,1)
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower_hsv = np.array([11, 43, 46])  # hsv中的白色过滤取值范围
        upper_hsv = np.array([25, 43, 255])
        mask = cv.inRange(frame_hsv, lower_hsv, upper_hsv)
        cv.imshow('video',frame)
        cv.imshow('video_mask',mask)
        c = cv.waitKey(10)
        if c == 27:
            break


# 通道分离与合并
def img_channel_split(img_path):
    img = cv.imread(img_path)
    img_b,img_g,img_r = cv.split(img)   # 将图像分离为3通道
    img_bg = cv.merge([img_b,img_g,img_r])    # 合并B和G通道
    img_bg[:,:,2] = 0
    img_br = cv.merge([img_b,img_g,img_r])    # 合并B和R通道
    img_br[:,:,1] = 0
    img_gr = cv.merge([img_b,img_g,img_r])    # 合并G和R通道
    img_gr[:,:,0] = 0
    cv.imwrite('img/素材/saonv_split_bg.jpg', img_bg)
    cv.imwrite('img/素材/saonv_split_br.jpg', img_br)
    cv.imwrite('img/素材/saonv_split_gr.jpg', img_gr)
    cv.imshow('saonv_bg',img_bg)
    cv.imshow('saonv_br',img_br)
    cv.imshow('saonv_gr',img_gr)

if __name__ == '__main__':
    #img_color_depart('./img/saonv.jpg','./img/saonv_reverse.jpg')
    #video_color_depart()
    img_channel_split('img/素材/saonv.jpg')
    cv.waitKey(0)
    cv.destroyAllWindows()  # 关闭窗口