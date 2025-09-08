# coding=utf-8

""""
参考：https://zhuanlan.zhihu.com/p/158769096 的思想，我没搞懂，结果跑通
将GDAL更改为rasterio
"""

import os
import rasterio
import numpy as np
import datetime
import math
# import sys
import pickle
import torch
from other.data_utils.data_preprocess import data_pre
from trainer.trainer import base_trainer

os.environ['PROJ_LIB'] = r'C:\ProgramData\anaconda3\envs\PU\Library\share\proj'
os.environ['GDAL_DATA'] = r'C:\ProgramData\anaconda3\envs\PU\Library\share'


def ReadRaster(file):
    """
    读取遥感影像数据.  parameters——raster:遥感影像的路径
    return——image：遥感影像包含的array，src_meta：遥感影像的坐标
    注意：rasterio读取数据结果为 通道数在最前面。
    """
    with rasterio.open(file) as src:
        if src == None:
            print(file + "文件无法打开")
        image = src.read()
        if image.shape[0] == 1:
            image = np.squeeze(image)
        src_meta = src.meta
    return image, src_meta


def TifCroppingArray(img, block_w, block_h, area_perc):
    """
    tif裁剪（tif像素数据，裁剪边长）
    block_w：图像块宽度，即行方向上像素个数，即列数； block_h：图像块高度，即列方向上像素个数，即行数
    """
    # 计算SideLength,即
    SideLength = int((1 - math.sqrt(area_perc)) * block_w / 2)
    #  裁剪链表
    TifArrayReturn = []
    #  列上图像块数目 即行数
    ColumnNum = int((img.shape[1] - SideLength * 2) / (block_w - SideLength * 2))
    #  行上图像块数目 即列数
    RowNum = int((img.shape[2] - SideLength * 2) / (block_h - SideLength * 2))
    for i in range(ColumnNum):
        TifArray = []
        for j in range(RowNum):
            cropped = img[:, i * (block_w - SideLength * 2): i * (block_w - SideLength * 2) + block_w,
                      j * (block_h - SideLength * 2): j * (block_h - SideLength * 2) + block_h]
            TifArray.append(cropped)
        TifArrayReturn.append(TifArray)
    #  考虑到行列会有剩余的情况，向前裁剪一行和一列
    #  向前裁剪最后一列
    for i in range(ColumnNum):
        cropped = img[:, i * (block_h - SideLength * 2): i * (block_h - SideLength * 2) + block_h,
                  (img.shape[2] - block_h): img.shape[2]]
        TifArrayReturn[i].append(cropped)
    #  向前裁剪最后一行
    TifArray = []
    for j in range(RowNum):
        cropped = img[:, (img.shape[1] - block_w): img.shape[1],
                  j * (block_w - SideLength * 2): j * (block_w - SideLength * 2) + block_w]
        TifArray.append(cropped)
    #  向前裁剪右下角
    cropped = img[:, (img.shape[1] - block_w): img.shape[1], (img.shape[2] - block_h): img.shape[2]]
    TifArray.append(cropped)
    TifArrayReturn.append(TifArray)
    #  列上的剩余数
    ColumnOver = (img.shape[1] - SideLength * 2) % (block_w - SideLength * 2) + SideLength
    #  行上的剩余数
    RowOver = (img.shape[2] - SideLength * 2) % (block_h - SideLength * 2) + SideLength
    return TifArrayReturn, RowOver, ColumnOver


def Result(shape, TifArray, npyfile, block_w, area_perc, RowOver, ColumnOver):
    """
    获得结果矩阵
    """
    RepetitiveLength = int((1 - math.sqrt(area_perc)) * block_w / 2)

    result = np.zeros(shape, np.uint8)
    #  j来标记行数
    j = 0
    for i, img in enumerate(npyfile):
        #  最左侧一列特殊考虑，左边的边缘要拼接进去
        if (i % len(TifArray[0]) == 0):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if (j == 0):
                result[0: block_w - RepetitiveLength, 0: block_w - RepetitiveLength] = img[
                                                                                       0: block_w - RepetitiveLength,
                                                                                       0: block_w - RepetitiveLength]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif (j == len(TifArray) - 1):
                #  原来错误的
                #result[shape[0] - ColumnOver : shape[0], 0 : 512 - RepetitiveLength] = img[0 : ColumnOver, 0 : 512 - RepetitiveLength]
                #  后来修改的
                result[shape[0] - ColumnOver - RepetitiveLength: shape[0], 0: block_w - RepetitiveLength] = img[
                                                                                                            block_w - ColumnOver - RepetitiveLength: block_w,
                                                                                                            0: block_w - RepetitiveLength]
            else:
                result[j * (block_w - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                        block_w - 2 * RepetitiveLength) + RepetitiveLength,
                0:block_w - RepetitiveLength] = img[RepetitiveLength: block_w - RepetitiveLength,
                                                0: block_w - RepetitiveLength]
        #  最右侧一列特殊考虑，右边的边缘要拼接进去
        elif (i % len(TifArray[0]) == len(TifArray[0]) - 1):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if (j == 0):
                result[0: block_w - RepetitiveLength, shape[1] - RowOver: shape[1]] = img[0: block_w - RepetitiveLength,
                                                                                      block_w - RowOver: block_w]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif (j == len(TifArray) - 1):
                result[shape[0] - ColumnOver: shape[0], shape[1] - RowOver: shape[1]] = img[
                                                                                        block_w - ColumnOver: block_w,
                                                                                        block_w - RowOver: block_w]
            else:
                result[j * (block_w - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                        block_w - 2 * RepetitiveLength) + RepetitiveLength,
                shape[1] - RowOver: shape[1]] = img[RepetitiveLength: block_w - RepetitiveLength,
                                                block_w - RowOver: block_w]
            #  走完每一行的最右侧，行数+1
            j = j + 1
        #  不是最左侧也不是最右侧的情况
        else:
            #  第一行的要特殊考虑，上边的边缘要考虑进去
            if (j == 0):
                result[0: block_w - RepetitiveLength,
                (i - j * len(TifArray[0])) * (block_w - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                    TifArray[0]) + 1) * (block_w - 2 * RepetitiveLength) + RepetitiveLength
                ] = img[0: block_w - RepetitiveLength, RepetitiveLength: block_w - RepetitiveLength]
            #  最后一行的要特殊考虑，下边的边缘要考虑进去
            elif (j == len(TifArray) - 1):
                result[shape[0] - ColumnOver: shape[0],
                (i - j * len(TifArray[0])) * (block_w - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                    TifArray[0]) + 1) * (block_w - 2 * RepetitiveLength) + RepetitiveLength
                ] = img[block_w - ColumnOver: block_w, RepetitiveLength: block_w - RepetitiveLength]
            else:
                result[j * (block_w - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                        block_w - 2 * RepetitiveLength) + RepetitiveLength,
                (i - j * len(TifArray[0])) * (block_w - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                    TifArray[0]) + 1) * (block_w - 2 * RepetitiveLength) + RepetitiveLength,
                ] = img[RepetitiveLength: block_w - RepetitiveLength, RepetitiveLength: block_w - RepetitiveLength]
    return result


def get_preprocessing(mean_file, std_file, img):
    """mean_file: 存储mean值的txt文件; std_file:存储std值的txt文件"""
    with open(mean_file, 'r', encoding='utf-8') as f:
        imgset_mean = np.array([line.strip('\n') for line in f]).astype('float32')
        # txt文件中只能存储str，这里读取进来数据类型也就是str，因而必须转换成数字才能计算
        # mean和std 肯定包括小数,因而这里读取后只能将数据类型转为float,而输入模型的数据类型
        # 默认是float32.
    with open(std_file, 'r', encoding='utf-8') as f:
        imgset_std = np.array([line.strip('\n') for line in f]).astype('float32')
    imgset_mean = np.expand_dims(np.expand_dims(imgset_mean, axis=1), axis=2)
    imgset_std = np.expand_dims(np.expand_dims(imgset_std, axis=1), axis=2)
    # 读取进来的img的shape为(height,width,band/frame), mask的shape为(height,width,band).
    # img的shape的格式由(height,width,band/frame)改为(band/frame,height,width).
    # mask的shape的格式由(height,width,band)改为(band,height,width)
    img = (img - imgset_mean) / imgset_std
    # 我得到的数据的shape为(16,256,256)，而imgset_mean的shape为(16,),imgset_stdd的的shape为(16,),
    # 可以直接按照numpy数组计算中的广播来计算
    img = img.astype('float32')

    return img


if __name__ == '__main__':

    #  获取当前时间
    starttime = datetime.datetime.now()
    block_w = 512
    block_h = 512
    area_perc = 0.95  # 预测区域，或者说非重叠区域

    #
    TifPath_7 = r"D:\MasterProgram\paper2\GIS\L3\Image\Original"
    tif_name7 = "DP07_PMS_20230810145244_200185191_104_0011_001_L3C_PSH.tif"
    ex_name7 = "ocsvm"
    TifPath_7 = os.path.join(TifPath_7, tif_name7)

    "模型权重所在路径"
    # 可以是多个，存储在list中
    model_paths = [r"D:\MasterProgram\paper2\MyNET\log\Compare\ocsvm\ex1_20250124-031600\best_f1_at_epoch28.model"]

    ResultPath = os.path.join(r"D:\MasterProgram\paper2\GIS\L3\Compare",
                              ex_name7 + str(area_perc) + ".tif")

    # 读取tif为numpy array
    img_7, src_meta = ReadRaster(TifPath_7)

    TifArray_7, RowOver, ColumnOver = TifCroppingArray(img_7, block_w, block_h, area_perc)

    "定义自己的model"
    "配置文件,我这里只用配置文件配置了模型的参数,其实也可以用来配置所有参数"


    predicts = []
    for i in range(len(TifArray_7)):
        print(i,'/', len(TifArray_7))
        for j in range(len(TifArray_7[0])):
            image_7 = TifArray_7[i][j]  # 4x512x512
            # print('image_7.shape', image_7.shape)
            image_concat = data_pre(image_7)

            pred = np.zeros((1, block_w, block_h))
            for model_path in model_paths:
                # 策略2：使用一个模型只做一次预测,只对原图做预测
                f2 = open(model_path, 'rb')
                s2 = f2.read()
                model = pickle.loads(s2)
                with torch.no_grad():
                    reshaped_array = image_concat.transpose(1, 2, 0)  # 将第二个维度移到最后
                    # 保留最后一维并转为一维列表
                    features = reshaped_array.reshape(-1, 6).tolist()
                    pred_pro = model.predict(features).reshape(512, -1)

                    pred = pred + pred_pro
            pred=pred/len(model_paths)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            pred = pred.astype(np.uint8)
            pred = pred.squeeze()
            predicts.append(pred)

    TifArray_8, TifArray_9, TifArray_10 = None, None, None

    # 保存结果predictspredicts
    result_shape = (img_7.shape[1], img_7.shape[2])
    result_data = Result(result_shape, TifArray_7, predicts, block_w, area_perc, RowOver, ColumnOver)
    src_meta.update({'count': 1, 'dtype': rasterio.uint8, 'compress': 'lzw', 'nodata': 0})

    with rasterio.open(ResultPath, 'w', **src_meta) as dst:
        dst.write(result_data, 1)


    #  获取当前时间
    endtime = datetime.datetime.now()
    print("模型预测完毕,目前耗时间: " + str((endtime - starttime).seconds) + "s")