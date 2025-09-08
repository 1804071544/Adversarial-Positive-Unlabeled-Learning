import numpy as np
import torch
from parameters import config


def _np_mean_std_normalize(image, mean, std):
    """
    Args:
        image: 3-D array of shape [height, width, channel]
        mean:  a list or tuple or ndarray
        std: a list or tuple or ndarray
    Returns:
    """
    if not isinstance(mean, np.ndarray):
        mean = np.array(mean, np.float32)
    if not isinstance(std, np.ndarray):
        std = np.array(std, np.float32)
    shape = [1] * image.ndim
    shape[-1] = -1
    return (image - mean.reshape(shape)) / std.reshape(shape)


def _th_mean_std_normalize(image, mean, std):
    """ this version faster than torchvision.transforms.functional.normalize
    Args:
        image: 3-D or 4-D array of shape [batch (optional) , height, width, channel]
        mean:  a list or tuple or ndarray
        std: a list or tuple or ndarray
    Returns:
    """
    shape = [1] * image.dim()
    shape[-1] = -1
    mean = torch.tensor(mean, requires_grad=False).reshape(*shape)
    std = torch.tensor(std, requires_grad=False).reshape(*shape)

    return image.sub(mean).div(std)


def mean_std_normalize(image, mean, std):
    """
    Args:
        image: 3-D array of shape [height, width, channel]
        mean:  a list or tuple
        std: a list or tuple
    Returns:
    """
    # print("Data normalization..........")
    if isinstance(image, np.ndarray):
        return _np_mean_std_normalize(image, mean, std)
    elif isinstance(image, torch.Tensor):
        return _th_mean_std_normalize(image, mean, std)
    else:
        raise ValueError('The type {} is not support'.format(type(image)))


def divisible_pad(image_list, size_divisor=128):
    # 确保图像的维度可以被整除
    """
    Args:
        image_list: a list of images with shape [channel, height, width]
        size_divisor: int
        to_tensor: whether to convert to tensor
    Returns:
        blob: 4-D ndarray of shape [batch, channel, divisible_max_height, divisible_max_height]
    """
    max_shape = np.array([im.shape for im in image_list]).max(axis=0)

    max_shape[1] = int(np.ceil(max_shape[1] / size_divisor) * size_divisor)
    max_shape[2] = int(np.ceil(max_shape[2] / size_divisor) * size_divisor)

    out = np.zeros([len(image_list), max_shape[0], max_shape[1], max_shape[2]], np.float32)

    for i, resized_im in enumerate(image_list):
        out[i, :, 0:resized_im.shape[1], 0:resized_im.shape[2]] = torch.from_numpy(resized_im)

    return out


def minibatch_sample(positive_train_indicator: np.ndarray, unlabeled_train_indicator: np.ndarray, sub_num_iter,
                     seed):
    rs = np.random.RandomState(seed)
    # 获取索引
    positive_train_indicator_inds = np.where(positive_train_indicator.ravel() == 1)[0]
    positive_minibatch_size = int(len(positive_train_indicator_inds) / sub_num_iter)
    unlabeled_train_indicator_inds = np.where(unlabeled_train_indicator.ravel() == 1)[0]
    unlabeled_minibatch_size = int(len(unlabeled_train_indicator_inds) / sub_num_iter)

    rs.shuffle(positive_train_indicator_inds)  # 打乱顺序
    rs.shuffle(unlabeled_train_indicator_inds)  # 打乱顺序

    positive_train_inds_list = []
    unlabeled_train_inds_list = []
    cnt = 0
    while True:
        positive_train_inds = np.zeros_like(positive_train_indicator).ravel()
        unlabeled_train_inds = np.zeros_like(unlabeled_train_indicator).ravel()

        positive_left = cnt * positive_minibatch_size
        positive_right = min((cnt + 1) * positive_minibatch_size, len(positive_train_indicator_inds))
        if positive_left < positive_right:
            positive_fetch_inds = positive_train_indicator_inds[positive_left:positive_right]
            positive_train_inds[positive_fetch_inds] = 1  # 将展开的inds中选中的位置置为1
            positive_train_inds_list.append(positive_train_inds.reshape(positive_train_indicator.shape))

        unlabeled_left = cnt * unlabeled_minibatch_size
        unlabeled_right = min((cnt + 1) * unlabeled_minibatch_size, len(unlabeled_train_indicator_inds))
        if unlabeled_left < unlabeled_right:
            unlabeled_fetch_inds = unlabeled_train_indicator_inds[unlabeled_left:unlabeled_right]
            unlabeled_train_inds[unlabeled_fetch_inds] = 1
            unlabeled_train_inds_list.append(unlabeled_train_inds.reshape(unlabeled_train_indicator.shape))

        cnt += 1
        if positive_train_inds.sum() == 0 or unlabeled_train_inds.sum() == 0:
            dataset_length = min(len(positive_train_inds_list), len(unlabeled_train_inds_list))
            return positive_train_inds_list[:dataset_length], unlabeled_train_inds_list[:dataset_length]


def cal_ndvi(image):
    # R G B NIR
    ndvi = (image[:, :, 3] - image[:, :, 0]) / (image[:, :, 3] + image[:, :, 0])
    ndvi = np.nan_to_num(ndvi, nan=0)
    return ndvi


def cal_ndwi(image):
    # R G B NIR
    ndwi = (image[:, :, 1] - image[:, :, 3]) / (image[:, :, 1] + image[:, :, 3])
    ndwi=np.nan_to_num(ndwi, nan=0)
    return ndwi


def data_pre(image):
    image = image.transpose((1, 2, 0))
    ndvi = cal_ndvi(image)  # 计算指数
    ndwi = cal_ndwi(image)
    im_cmean = config['dataset']['train']['im_cmean']
    im_cstd = config['dataset']['train']['im_cstd']
    image = mean_std_normalize(image, im_cmean, im_cstd)  # 先对RGB进行归一化
    image = np.concatenate((image, np.expand_dims(ndvi, axis=2)), axis=2)  # 合并指数和光谱
    image = np.concatenate((image, np.expand_dims(ndwi, axis=2)), axis=2)
    image = image.transpose((2, 0, 1)).astype(np.float32)
    return image
