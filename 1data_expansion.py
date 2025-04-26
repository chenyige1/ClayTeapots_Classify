"""
Created on 2025/04/24
@author: cyg
opencv数据扩充：添加 亮度，噪声，旋转，裁剪
"""

import os
import shutil
import cv2
import numpy as np


def contrast_brightness_image(src1, a, g, path_out):
    '''
        色彩增强（通过调节对比度和亮度）
    '''
    h, w, ch = src1.shape  # 获取shape的数值，height和width、通道
    # 新建全零图片数组src2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
    src2 = np.zeros([h, w, ch], src1.dtype)
    # addWeighted函数说明:计算两个图像阵列的加权和
    dst = cv2.addWeighted(src1, a, src2, 1 - a, g)
    cv2.imwrite(path_out, dst)


def gauss_noise(image, path_out_gauss, mean=0, var=0.01):
    '''
    添加高斯噪声
    mean : 均值
    var : 方差
    '''
    # 将图像转换为浮点型并归一化到 [0, 1]
    image = np.array(image, dtype=float) / 255.0
    # 生成高斯噪声
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    # 将噪声添加到图像上
    out = image + noise
    # 确保输出的像素值在 [0, 1] 范围内
    out = np.clip(out, 0, 1)
    # 转换回 [0, 255] 的整数类型
    out = np.uint8(out * 255)
    cv2.imwrite(path_out_gauss, out)


def mirror(image, path_out_mirror):
    '''
        水平镜像
    '''
    h_flip = cv2.flip(image, 1)
    cv2.imwrite(path_out_mirror, h_flip)


def resize(image, path_out_large):
    '''
        放大两倍
    '''
    height, width = image.shape[:2]
    large = cv2.resize(image, (2 * width, 2 * height))
    cv2.imwrite(path_out_large, large)


def rotate(image, path_out_rotate):
    '''
        旋转
    '''
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 20, 1)
    dst = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(path_out_rotate, dst)


def shear(image, path_out_shear):
    '''
        剪切
    '''
    height, width = image.shape[:2]
    cropped = image[int(height / 5):height, int(width / 5):width]
    cv2.imwrite(path_out_shear, cropped)


def main():
    image_path = './image/1'
    image_out_path = './image/1_test'
    # image_path = './dataset/Square'
    # image_out_path = './datasets/Square'
    if not os.path.exists(image_out_path):
        os.mkdir(image_out_path)
    list = os.listdir(image_path)
    print(list)
    print("----------------------------------------")
    print("The original data path:" + image_path)
    print("The original data set size:" + str(len(list)))
    print("----------------------------------------")

    imageNameList = [
        '_color.jpg',
        '_gauss.jpg',
        '_rotate.jpg',
        '_shear.jpg',
        '.jpg']
    for i in range(0, len(list)):
        path = os.path.join(image_path, list[i])
        out_image_name = os.path.splitext(list[i])[0]
        for j, n in enumerate(imageNameList):
            path_out = os.path.join(
                image_out_path, out_image_name + imageNameList[j])
            image = cv2.imread(path)
            if n == "_color.jpg":
                contrast_brightness_image(image, 1.5, 10, path_out)
            elif n == "_gauss.jpg":
                gauss_noise(image, path_out)
            elif n == "_mirror.jpg":
                mirror(image, path_out)
            elif n == "_large.jpg":
                resize(image, path_out)
            elif n == "_rotate.jpg":
                rotate(image, path_out)
            elif n == "_shear.jpg":
                shear(image, path_out)
            else:
                shutil.copy(path, path_out)
        print(out_image_name + "success！", end='\t')
    print("----------------------------------------")
    print("The data augmention path:" + image_out_path)
    outlist = os.listdir(image_out_path)
    print("The data augmention sizes:" + str(len(outlist)))
    print("----------------------------------------")
    print("Rich sample for:" + str(len(outlist) - len(list)))


if __name__ == '__main__':
    main()
