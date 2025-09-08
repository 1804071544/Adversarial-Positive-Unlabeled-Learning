#!/.conda/envs/learn python
# -*- coding: utf-8 -*-

"""
样本制作的第一步
根据点shp文件，以各个点为中心，将周围的小块影像裁剪出来
注意：*必须保证原始影像和点shp文件的投影一致
注意：在制作点文件时候（比如在QGIS中），要添加size field字段，因为裁剪多大的影像块（影像块的像素个数）是由每个feature的size field中的值决定的。
注意：在制作点文件时，还要注意id field中的值，因为我们裁剪出的影像块的命名对应于每个feature的id值。
总结：将裁剪出来的影像块的尺寸和命名都由shpfile中的layer中的每个feature的id field和size field来决定，优点是方便文件的管理，不用再将不同尺寸
影像块对应的呃点再分门别类存放。缺点是不够灵活（想裁剪多大尺寸在shp文件外面自己定义更加灵活）。
"""

from osgeo import gdal
from osgeo import ogr
import fnmatch
import os
import sys
import numpy as np


def write_img(out_path, im_proj, im_geotrans, im_data):
    """输出影像
    目前仅支持tif格式
        out_path: 输出路径
        im_proj: 输出图像的仿射矩阵
        im_geotrans 输出图像的空间参考
        im_data 输出图像的数据，以np.array格式存储
    """
    # 判断数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 计算波段数
    if len(im_data.shape) > 2:  # 多波段
        im_bands, im_height, im_width = im_data.shape
    else:  # 单波段
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建新影像
    driver = gdal.GetDriverByName("GTiff")
    new_dataset = driver.Create(
        out_path, im_width, im_height, im_bands, datatype, options=['COMPRESS=LZW', 'BIGTIFF=YES'])
    # options=['COMPRESS=LZW', 'BIGTIFF=YES'] 操作来自
    # https://blog.csdn.net/dou3516/article/details/105094323?spm=1001.2014.3001.5506
    new_dataset.SetGeoTransform(im_geotrans)
    new_dataset.SetProjection(im_proj)
    if im_bands == 1:
        new_dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            new_dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del new_dataset


def clip(out_tif_name, sr_img, point_shp, cut_cnt, id: str = "id", size: str = "size"):
    """裁剪主函数
    根据点shp从原始影像中裁剪出小块影像用来作样本
        out_tif_name: 输出小tif影像的不完整路径 后面的_n.tif在该函数中补充
        sr_img: 原始影像的完整路径
        point_shp 点shp的完整路径
        id: str = "id"。裁剪输出的影像块的命名根据每个feature的id值来定.
        size: str = "size".裁剪的影像块的尺寸由每个feature的size值来定.
    """
    # 读取原始影像
    im_dataset = gdal.Open(sr_img)
    if im_dataset == None:
        print('open sr_img false')
        sys.exit(1)  # 0为正常，1~127为异常
    im_geotrans = im_dataset.GetGeoTransform()  # GDAL(python) 之GeoTransform https://blog.csdn.net/RSstudent/article/details/108732571
    im_proj = im_dataset.GetProjection()
    im_width = im_dataset.RasterXSize
    im_height = im_dataset.RasterYSize

    # 读取样本点矢量文件
    shp_dataset = ogr.Open(point_shp)
    if shp_dataset == None:
        print('open shapefile false')
        sys.exit(1)
    layer = shp_dataset.GetLayer()
    # extent = layer.GetExtent()可读出 layer 的上下左右边界
    point_proj = layer.GetSpatialRef()

    # 读取layer中的某一要素feature  feature = layer.GetFeature() fid = feat.GetField('id')
    # 详见教程https://www.osgeo.cn/python-gdal-utah-tutorial/ch02.html
    # 按顺序读取layer中的feature，  feat = layer.GetNextFeature()
    # 循环遍历layer中的所有feature
    # while feat:
    #     feat = layer.GetNextFeature()

    feature = layer.GetNextFeature()
    while feature:
        geom = feature.GetGeometryRef()  # 提取feature的几何形状 geom = feat.GetGeometryRef() geom.GetX() geom.GetY()
        name_id = int(feature.GetField(id))
        # if name_id not in [60]:
        #     continue
        data_size = int(feature.GetField(size))
        adata_size = int(data_size / 2)
        geoX = float(geom.GetX())  # feature的左上角x坐标
        geoY = float(geom.GetY())  # feature的左上角y坐标
        g0 = float(
            im_geotrans[0])  # GDAL(python) 之 GeoTransform https://blog.csdn.net/RSstudent/article/details/108732571
        g1 = float(im_geotrans[1])  # 在GDAL for Python中，GeoTransform是一个六个元素的元组
        g2 = float(im_geotrans[2])  # 六个参数分别为：左上角x坐标，水平分辨率，旋转参数，左上角y坐标，旋转参数，竖直分辨率。
        g3 = float(im_geotrans[3])  # 如 (486892.5, 15.0, 0.0, 4105507.5, 0.0, -15.0)
        g4 = float(im_geotrans[4])  # 满足如下关系式：Xgeo = GT(0) + Xpixel*GT(1) + Yline*GT(2)
        g5 = float(im_geotrans[5])
        # 上面的 im_geotrans 是前面遥感影像的 位置信息，而geoX和geoY是shp矢量位置信息
        # 下面是 坐标系 和 栅格array行列数 的换算  是计算feature的坐标在栅格坐标中的对应行列位置
        x = (geoX * g5 - g0 * g5 - geoX * g2 + g3 * g2) / (g1 * g5 - g4 * g2)  # 本质上是（geoX-g0）/g1 即 坐标/分辨率=栅格array行列数
        y = (geoY - g3 - geoX * g4) / g5  # 本质上是（geoY-g3）/g5 即 坐标/分辨率=栅格array行列数

        x, y = int(x), int(y)

        a1 = x - adata_size
        a2 = y - adata_size
        a3 = x + adata_size
        a4 = y + adata_size
        if a1 > 0 and a2 > 0 and a3 > 0 and a4 > 0 and a3 < im_width and a4 < im_height:
            cut_cnt = cut_cnt + 1
            geoX2 = g0 + g1 * a1 + g2 * a2  # 将 feature 在栅格中对应的行列位置换算回坐标位置
            # print('geoX2',geoX2)
            geoY2 = g3 + g4 * a1 + g5 * a2  # 将 feature 在栅格中对应的行列位置换算回坐标位置
            # print('geoY2', geoY2)
            im_data = im_dataset.ReadAsArray(a1, a2, data_size, data_size)
            im_geotrans_list = list(im_geotrans)
            im_geotrans_list[0] = geoX2  # 替换原影像坐标im_geotrans为feature对应的坐标
            im_geotrans_list[3] = geoY2  # 替换原影像坐标im_geotrans为feature对应的坐标
            # strname = out_tif_name + '_' + str(cut_cnt) + '.tif'
            if name_id < 10:
                str_name = out_tif_name + str(cut_cnt)+'_' + '00{}'.format(name_id) + '.tif'
            elif 10 <= name_id < 100:
                str_name = out_tif_name + str(cut_cnt)+'_' + '0{}'.format(name_id) + '.tif'
            else:
                str_name = out_tif_name + str(cut_cnt)+'_' + str(name_id) + '.tif'
            write_img(str_name, im_proj, im_geotrans_list, im_data)
            print('文件写入成功')
        feature.Destroy()  # 释放内存 feature.Destroy()
        feature = layer.GetNextFeature()  # 按顺序读取layer中的下一个feature

    shp_dataset.Destroy()  # 关闭数据源，相当于文件系统操作中的关闭文件 dataSource.Destroy()
    return cut_cnt


# 防止GDAL报ERROR4错误 gdal_gata文件夹需要相应配置
# os.environ['GDAL_DATA'] = r'C:\Users\75198\.conda\envs\learn\Lib\site-packages\GDAL-2.4.1-py3.6-win-amd64.egg-info\gata-data'

# sr_image_path = r"G:\manas_class\project_manas\0-src_img" #原始影像
# print(sr_image_path)
# point_shp = r"F:\grass\0-point_sample\New_Shapefile.shp" #中心点point文件

"Wa 第一次样本选择"
# out_path = r"D:\study_Irrigation\sampleRaster\Wa_label_val_preview"
# point_shp = r"D:\study_Irrigation\shp\Wa_sample\point_utm\WaIrrPonit_val.shp"
# sr_image_path = r"D:\study_Irrigation\Raw_label"

"Ca 第一次样本选择"
point_shp = r"D:\MasterProgram\Graduation_thesis\paper2\GIS\L3\PreTraining\shp\Train_Point.shp"
sr_image_path = r"D:\MasterProgram\Graduation_thesis\paper2\GIS\L3\PreTraining\Label\eye\tif"
out_path = r"D:\MasterProgram\Graduation_thesis\paper2\GIS\L3\PreTraining\Label"

# datasize = 768
# img_type = '*.dat' #原始影像类型 不可漏*.
img_type = '*.tif'
# output_prefix = 'manas_grass_add' #输出小块影像文件名的前缀
# output_prefix = 'WaIrr'
output_prefix = 'CaIrr'

if not os.path.exists(out_path):
    os.mkdir(out_path)

# 过滤出原始影像
sr_img_list = fnmatch.filter(os.listdir(sr_image_path), img_type)  # tif还是TIF注意区别
# sr_img_list = ["Wa_11N_C1.tif"]
# sr_img_list = ["label.tif"]

# adatasize = int(datasize / 2)
print(sr_img_list)

cnt = 0
cut_cnt = 0
for sr_img in sr_img_list:
    cnt = cnt + 1
    shp_name, extension = os.path.splitext(sr_img)
    sr_img = sr_image_path + '/' + shp_name + img_type[1:]
    print(sr_img)
    out_tif_name = out_path + '/' + output_prefix  # 改输出编号
    print('start clip image', cnt)
    cut_cnt = clip(out_tif_name, sr_img, point_shp, cut_cnt)
    print('clip image', cnt, 'done')
print('Finish!')
