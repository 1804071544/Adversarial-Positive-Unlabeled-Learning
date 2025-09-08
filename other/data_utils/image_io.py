from osgeo import gdal
import numpy as np
import torch
from other.data_utils.data_preprocess import mean_std_normalize


def read_ENVI(filepath):
    dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    data = dataset.ReadAsArray(0, 0, cols, rows)
    if len(data.shape) == 3:
        data = data.transpose((1, 2, 0))
    return data


def read_ENVI_Validation(filepath):
    dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    data = dataset.ReadAsArray(0, 0, cols, rows)
    if len(data.shape) == 3:
        data = data.transpose((1, 2, 0))
    image = data
    im_cmean = image.reshape((-1, image.shape[-1])).mean(axis=0)
    im_cstd = image.reshape((-1, image.shape[-1])).std(axis=0)
    image = mean_std_normalize(image, im_cmean, im_cstd)
    image = image.transpose(2, 0, 1)
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    image = image.astype(np.float32)
    image = torch.from_numpy(image)
    return image
