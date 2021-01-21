# OpenCV学习之旅
## 一. 图像处理的基础使用
### 1.环境的配置与opencv的初识
**OpenCV** 应用领域

- 1、人机互动
- 2、物体识别
- 3、图像分割
- 4、人脸识别
- 5、动作识别
- 6、运动跟踪
- 7、机器人
- 8、运动分析
- 9、机器视觉
- 10、结构分析
- 11、汽车安全驾驶

 【opencv库安装】
> pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python

==*注：opencv库在导入时应用import cv2==

### 2.opencv中的helloworld

```python
import cv2 as cv

src = cv.imread('./img/saonv.jpg')  # 载入图片资源
cv.namedWindow('show_img')  # 给窗口命名
cv.imshow('show_img',src)   # 将图像展示在已命名的窗口上
cv.waitKey(0)   # 停留的时间,单位为ms,0即无限呈现
cv.destroyWindow('show_img')    # 关闭窗口
```
**基础库函数讲解**

- [cv2.imread(filepath,flags=None)]--读取图片资源
  - filepath：图像路径【支持的图像格式很多，但不支持读取内存较大的图像】
  - flags: 读取图像方式
  	- cv2.IMREAD_COLOR：默认参数，读入一副彩色图片，忽略alpha通道
  	- cv2.IMREAD_GRAYSCALE：读入灰度图片
  	- cv2.IMREAD_UNCHANGED：顾名思义，读入完整图片，包括alpha通道
- [cv2.imshow(windname,mat)]--展示图片在窗口上
  - windname：窗口名【要展示的窗口名，需要用cv2.namewindow()给窗口命名后才能使用】 
  - mat:呈现的图像【cv2.imread()的返回值】
- [cv2.imwrite(filename, img, params=None)] --存储图像
  - filename:存储文件路径
  - img:存储的图像
  - params:压缩级别，默认为3 
- [cv2.cvtColor(src, code, dst=None, dstCn=None)] --图像颜色空间转换
  - src:图像资源
  - code: 转换形式
    - cv2.COLOR_RGB2GRAY:灰度化:彩色图像转灰度图像
    - cv2.COLOR_GRAY2RGB:彩色化:灰度图像转彩色图像
    - 其他：cv2.COLOR_{X}2{Y}，X，Y=RGB,BGR,GRAY,HSV,YCrCb,XYZ,Lab,Luv,HLS
- [cv2.resize(src, dsize, dst=None, fx=None, fy=None, interpolation=None)] --图像缩放 
  - src:原始图像
  - dsize:新图像尺寸
  - dst:新图像
  - fx:水平方向比例因子
  - fy:垂直方向比例因子
  - interpolation:插值方法
    - cv2.INTER_NEAREST:最近邻插值
    - cv2.INTER_LINEAR:双线性插值(默认)
    - cv2.INTER_AREA:使用像素区域关系进行重新采样，它可能是图像抽样的首选方法，因为它会产生无云纹理结果，但当图像缩放时，它类似INTER_NEAREST方法
    - cv2.INTER_CUBIC:4x4像素领域的双三次插值
    - cv2.INTER_LANCZOS4:8x8像素领域的Lanczos插值
- [cv2.flip(img,flipcode)]--图像翻转
  - img:图像资源
  - flipcode:翻转效果 
    - 1：水平翻转
    - 0：垂直翻转
    - -1：水平垂直翻转 
- [cv2.namewindow(name:str,flags=None)]--窗口命名
  - name:窗口名
  - flags:窗口的类型(大小,比例,屏幕绘制)
    - cv2.WINDOW_NORMAL：图片正常的尺寸
    - cv2.WINDOW_AUTOSIZE: 根据屏幕尺寸自动调整图片尺寸（默认）
    - cv2.WINDOW_FREERATIO：自由尺寸比例
    - cv2.WINDOW_KEEPRATIO：保持图像原有尺寸比例（默认）
    - cv2.WINDOW_GUI_NORMAL：原有绘制图像方法
    - cv2.WINDOW_GUI_EXPANDED：新增强绘制图像方法（默认）
### 3.获取图像的基础信息

```python
import cv2 as cv

def get_img_info(img):
	print(img.shape)
	print(img.size)
	print(dtype)
```

- [cv2,imread().shape]--获取图像尺寸信息
  - 返回tuple(高，宽，通道数)，其中3表示3位通道，即RGB
- [cv2.imread().size]--获取图片大小
  - 返回int(size)，size=高 * 宽 * 通道数
- [cv2.imread().dtype]--图像的字节类型
  - 返回type(dtype)，其中uint8即8位无字符字节
### 4.对图像进行像素取反
```python
def reverse_pixels(img):
    new_img = cv.bitwise_not(img)	## 像素值取反
    #new2_img = cv.bitwise_or(img,new_img)	## 两张图片像素值参与二进制的或运算
    #cv.imshow('saonv_or',new2_img)		## 结果为白色
   # new3_img = cv.bitwise_and(img,new_img)	## 两张图片像素值参与二进制的和运算
    #cv.imshow('saonv_and',new3_img)	## 结果为黑色
    cv.imwrite('./img/saonv_reverse.jpg',new_img)
    cv.imshow('saonv_reverse',new_img)
```

### 5.调用摄像头并呈现视屏
```python
import cv2 as cv

def use_capture_video():
    capture = cv.VideoCapture(0)  # 启用摄像头
    while(True):  # 不停读取摄像头捕捉到的图像并形成视屏
        ret,frame = capture.read()
        frame = cv.flip(frame,1)  # 镜像翻转
        cv.imshow('video',frame)
        c = cv.waitKey(1)	# 等待的时长
        if c == 27:
            break
```
### 6.opencv中的计时函数
```python
import cv2 as cv

def count_time(fn):
    def inner(img):
        start = cv.getTickCount()
        fn(img)
        end = cv.getTickCount()
        count = (end-start)/cv.getTickFrequency()
        print('函数[{}]耗时{:.5f} ms'.format(fn.__name__,count*1000))
    return inner

@count_time
funtion()
> 装饰器计时图像处理函数模板
```
### 7.用Numpy创建一个纯色图片
```python
# 创建一个纯色图片（3通道）
def create_img(RGB):
    img = np.zeros([450,450,3],np.uint8)
    img[:,:,0] = np.ones([450,450]) * RGB[0]   # B通道的值
    img[:,:,1] = np.ones([450,450]) * RGB[1]    # G通道的值
    img[:,:,2] = np.ones([450,450]) * RGB[2]    # R通道的值
    cv.imshow('show_RGB',img)

# 创建一个灰度纯色图片（单通道）
def create_grey_img(grey_degree):
    img = np.ones([250,250,1],np.uint8) * grey_degree
    cv.imshow('show_grey_img',img)
```
原理：

> - 用numpy.zeros创建一个[450,450,3]值全为0，数据类型为unit8的三维数组，即通道为3的高为450，宽为450像素图片
> - 将RGB通道的值分别全部赋予相对于RGB颜色的值
> - 用cv2.imshow()将图片展现出来（对于每张图片，每个像素点的值都是(R,G,B)(R:1~255,G:1~255,B:1~255)）
>
>  **注：opencv的RGB通道并非按R，G，B来赋值，而是B，G，R** 

==【基础函数讲解】==
- [(p_object, dtype=None, copy=True, order='K', subok=False, ndmin=0)]--创建一个数组
- p_object:要创建的数组
  - dtype:数组的字节类型，有uint8,uint16,int8,int16,float8,float16,float64(默认)等,
  - cpoy:数组是否可以复制,默认为True
  - order='K'/'A'/'C'/'F':在内存中的存储顺序，默认为'K'
  ![image-20201220142830100](https://cdn.jsdelivr.net/gh/MGod-monkey/OpenCV_StyudySelf@master/img/MarkDown/image-20201220142830100.png)
    - 'K'/'A'只有在copy=Ture时存储的顺序有区别，其他时候本质就是**行优先**还是**列优先**的顺序差别
  - subok=False:子类是否可传递
    - True:子类将被传递
    - False(默认);返回的数组被强制为基类数组 
  - ndmin=0:指定结果数组所应有的最小维度，默认为最小为0维度
- [array.reshape(shape)]--重置数组的规模
  - 如一个3x3的二维数组通过array.reshape([1,9])转换为一个1x9的一维数组
  -  **注:重置的新数组在数量上应与原有数组一致，否则会报错** 
- [numpy.zeros(shape, dtype=None, order='C')]--创建一个全为0的指定规模数组
  - shape:数组的规模
  - dtype:数组的字节类型，有uint8,uint16,int8,int16,float8,float16,float64(默认)等,
  -  order='C'/'F':在内存中的存储顺序,不影响
    - 该顺序定义是在内存中以行优先（C风格）还是列优先（Fortran风格）顺序存储多维数组
- [numpy.zeros(shape, dtype=None, order='C')]--创建一个全为1的指定规模数组
  - shape:数组的规模
  - dtype:数组的字节类型，有uint8,uint16,int8,int16,float8,float16,float64(默认)等,
  - order='C'/'F':在内存中的存储顺序,默认为'C'
    - 该顺序定义是在内存中以行优先（C风格）还是列优先（Fortran风格）顺序存储多维数组
- [array.fill(self,value)]--数组填充全为某个值
  - value:要填充的值
### 8.色彩空间与色彩分离
#### a.色彩空间
> 色彩学中，人们建立了多种色彩模型，以一维、二维、三维甚至四维空间坐标来表示某一色彩，这种坐标系统所能定义的色彩范围即色彩空间。
- RGB(Red,Green,Blue):

> ![image-20201220162411685](https://cdn.jsdelivr.net/gh/MGod-monkey/OpenCV_StyudySelf@master/img/MarkDown/image-20201220162411685.png)
- HSV(色相/饱和度/明度):

> ![image-20201220162536931](https://cdn.jsdelivr.net/gh/MGod-monkey/OpenCV_StyudySelf@master/img/MarkDown/image-20201220162536931.png)
- HSL(色相/亮度/明度)

> ![image-20201220162718097](https://cdn.jsdelivr.net/gh/MGod-monkey/OpenCV_StyudySelf@master/img/MarkDown/image-20201220162718097.png)

#### b.色彩分离
> 通过cv.inRange函数执行基本阈值操作,来对图像的特定色彩进行分离
> ![image-20201220163624569](https://cdn.jsdelivr.net/gh/MGod-monkey/OpenCV_StyudySelf@master/img/MarkDown/image-20201220163624569.png)
```python
# 图像色彩分离
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
    cv.imwrite('./img/saonv_mask.jpg', mask)
    cv.imshow('img_mask', mask)
    cv.imwrite('./img/saonv_reverse_mask.jpg', mask2)
    cv.imshow('img_reverse_mask', mask2)


# 视频色彩筛选
def video_color_depart():
    capture = cv.VideoCapture(0)
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv.flip(frame,1)
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower_hsv = np.array([0, 0, 221])  # hsv中的白色过滤取值范围
        upper_hsv = np.array([180, 30, 255])
        mask = cv.inRange(frame_hsv, lower_hsv, upper_hsv)
        cv.imshow('video_mask',mask)
        c = cv.waitKey(10)
        if c == 27:
            break

if __name__ == '__main__':
    img_color_depart('./img/saonv.jpg','./img/saonv_reverse.jpg')
    video_color_depart()
    cv.waitKey(0)
    cv.destroyAllWindows()  # 关闭窗口
```
> 通过cv2.split()对图像的通道进行分离
```python
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
    cv.imwrite('./img/saonv_split_bg.jpg',img_bg)
    cv.imwrite('./img/saonv_split_br.jpg',img_br)
    cv.imwrite('./img/saonv_split_gr.jpg',img_gr)
    cv.imshow('saonv_bg',img_bg)
    cv.imshow('saonv_br',img_br)
    cv.imshow('saonv_gr',img_gr)
```
### 9.图像中的逻辑运算

```python
# 两张图片的相加减运算
def lj_operation(img1_path,img2_path):
    img1 = cv.imread(img1_path)
    img2 = cv.imread(img2_path)
    img1_and_img2 = cv.add(img1,img2)   # 相加运算
    cv.imwrite('./img/img1_and_img2.jpg',img1_and_img2)
    cv.imshow('img_and',img1_and_img2)
    img1_sub_img2 = cv.subtract(img1,img2)  # 相减运算
    cv.imwrite('./img/img1_sub_img2.jpg', img1_sub_img2)
    cv.imshow('img_sub', img1_sub_img2)
    img2_sub_img1 = cv.subtract(img2, img1)  # 相减运算
    cv.imwrite('./img/img2_sub_img1.jpg', img2_sub_img1)
    cv.imshow('img2_sub', img2_sub_img1)
```

![相加](https://cdn.jsdelivr.net/gh/MGod-monkey/OpenCV_StyudySelf@master/img/MarkDown/image-20201220175553539.png)

![相减1](https://cdn.jsdelivr.net/gh/MGod-monkey/OpenCV_StyudySelf@master/img/MarkDown/image-20201220175739517.png)

![相减2](https://cdn.jsdelivr.net/gh/MGod-monkey/OpenCV_StyudySelf@master/img/MarkDown/image-20201220180041509.png)
==基础函数讲解==
==图像中的与或运算==

- [cv2.bitwise_not(src, dst=None, mask=None)]--图片像素值的取反运算
  - src:要处理的图像
  - dst:指定输出的新图像
  - mask:蒙版（8位通道数组），即不参与运算的部分
  - **注：输出像素值=255-原像素值**
- [cv2.bitwise_and(src1, src2, dst=None, mask=None)]--图像像素值的与运算
  - src1:目标图像
  - src2:源图像
  - dst:指定输出的新图像
  - mask:蒙版（8位通道数组），即不参与运算的部分
  - **注：当且仅当两个像素都大于零时，按位AND才为真，相与取较大值为结果**
- [cv2.bitwise_or(src1, src2, dst=None, mask=None)] --图像像素值的或运算
  - src1:目标图像
  - src2:源图像
  - dst:指定输出的新图像
  - mask:蒙版（8位通道数组），即不参与运算的部分
  - **注：如果两个像素中的任何一个大于零，则按位“或”为真，相或取较小值为结果**
- [cv2.bitwise_xor(src1, src2, dst=None, mask=None)]--图像像素值的异或运算
  - src1:目标图像
  - src2:源图像
  - dst:指定输出的新图像
  - mask:蒙版（8位通道数组），即不参与运算的部分
  - **注：当且仅当两个像素转化为二进制进行异或计算**
- [cv2.add(src1, src2, dst=None, mask=None, dtype=None)]--两个图像之间的像素值加法运算
  - src1:图像1
  - src2:图像2
  - dst:指定输出新图像
  - mask:蒙版(不参与运算部分)
  - dtype:字节类型，不给值时，以参与运算图像的dtype中最大的为输出图像的dtype
    - 当字节类型为uint8时,即像素值为0~255，当两个像素值相加结果超出规定的字节类型范围时，自动设置为最大或最小值
- [cv2.subtract(src1, src2, dst=None, mask=None, dtype=None)]--两个图像之间的像素值减法运算
  - src1:图像1
  - src2:图像2
  - dst:指定输出新图像
  - mask:蒙版(不参与运算部分)
  - dtype:字节类型，不给值时，以参与运算图像的dtype中最大的为输出图像的dtype
    - 当字节类型为uint8时,即像素值为0~255，当两个像素值相减结果超出规定的字节类型范围时，自动设置为最大或最小值
- > 参考博客：https://blog.csdn.net/zhouzongzong/article/details/93028651
### 10.提高图片的对比度和亮度
> 原理：创建一个与原图具有相同规模的零数组，再通过cv2.addweighted()将两张图片分别以不同的权重比叠加在一起
```python
# 提升图片的对比度和亮度
def update_light(img_path,d,l):	# d:对比度的值，l:亮度的值
    img = cv.imread(img_path)
    h,w,ch = img.shape
    blank = np.zeros([h,w,ch], img.dtype)   # 创建一个与图像相同规模的数组
    dst = cv.addWeighted(img,d,blank,1-d,l)
    cv.imshow('update_light',dst)
    cv.imshow('img',img)
    cv.imwrite('./img/lena_light.jpg',dst)
```
==基础函数解析==
- [cv2.addWeighted(src1, alpha, src2, beta, gamma, dst=None, dtype=None)]--图像叠加or图像混合加权实现
  - src1:图像1
  - alpha:图像1的权重
  - src2:图像2
  - beta:图像2的权重
  - dst:指定输出的新图像
  - gamma:对每个和的标量相加
  - dtype:可选的输出数组深度;当两个输入数组具有相同的深度时，可以将dtype设置为-1，这相当于src1.depth()
> 输出的图像像素值dst = src1 * alpha + src2 * beta + gamma

## 图像处理进阶操作
### 1.ROI与泛洪填充
> ROI（region of interest），感兴趣区域。机器视觉、图像处理中，从被处理的图像以方框、圆、椭圆、不规则多边形等方式勾勒出需要处理的区域，称为感兴趣区域，ROI。
> 学习参考博客：https://www.cnblogs.com/FHC1994/p/9033580.html
- 彩色图像填充
```python
#泛洪填充(彩色图像填充)
import cv2 as cv
import numpy as np
def fill_color_demo(img_path):
    copyImg = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros([h+2, w+2],np.uint8)   #mask必须行和列都加2，且必须为uint8单通道阵列
    #为什么要加2可以这么理解：当从0行0列开始泛洪填充扫描时，mask多出来的2可以保证扫描的边界上的像素都会被处理
    cv.floodFill(copyImg, mask, (220, 250), (0, 255, 255), (100, 100, 100), (50, 50 ,50), cv.FLOODFILL_FIXED_RANGE)
    cv.imshow("fill_color_demo", copyImg)
```


