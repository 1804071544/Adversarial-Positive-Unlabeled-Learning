from os import listdir
from other.data_utils.image_io import read_ENVI
import fnmatch
import os
import random
import albumentations as albu
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from other.data_utils.data_preprocess import mean_std_normalize, divisible_pad, minibatch_sample, cal_ndvi, cal_ndwi
from parameters import sentinel_use_flag, data_enhancement_flag


class TrainDataset(data.Dataset):
    def __init__(self, config, data_root_path, stuff=True):
        super(TrainDataset, self).__init__()
        ##########图像路径及文件名##########
        self.data_root_path = data_root_path

        self.train_flag = config['dataset']['train']['train_flag']  # 是否训练模式
        self.cls = config['dataset']['train']['ccls']
        self.sub_num_iter = config['dataset']['train']['sub_minibatch']
        self.num_train_samples_per_class = config['dataset']['train']['num_positive_train_samples']
        self.ratio = config['dataset']['train']['ratio']
        self.im_cmean = config['dataset']['train']['im_cmean']
        self.im_cstd = config['dataset']['train']['im_cstd']
        self._seed = 2333  # 随机数种子
        self._rs = np.random.RandomState(self._seed)
        # set list length = 9999 to make sure seeds enough
        self.seeds_for_minibatchsample = [e for e in self._rs.randint(low=2147483648, size=9999)]
        self.image = None
        self.sentinel_use_flag = sentinel_use_flag
        self.data_enhancement_flag = data_enhancement_flag
        # 读取目标文件下的所有tif文件
        self.image_path = os.path.join(self.data_root_path, 'Image', 'Train')
        self.gt_path = os.path.join(self.data_root_path, 'Label', 'Train')
        self.image_list = self.read_tif_list(self.image_path)
        self.gt_list = self.read_tif_list(self.gt_path)
        self.actual_batch_size = self.sub_num_iter + 1 if self.num_train_samples_per_class % self.sub_num_iter else self.sub_num_iter
        self.trans = self.get_training_augmentation()

    def get_training_augmentation(self):
        train_transform = [
            albu.HorizontalFlip(p=0.6),
            albu.VerticalFlip(p=0.6),
            albu.Transpose(p=0.6),
            albu.RandomRotate90(p=0.5)
        ]
        return albu.Compose(train_transform)

    def preset(self, index):
        self.image = None  # 清空上一次image
        self.mask = None
        index = max(0, index)  # 防止左侧索引溢出
        index = min(index, int(len(self.image_list) / 4) - 1)  # 防止右侧索引溢出
        img_list = fnmatch.filter(self.image_list, str(index) + '*')  # 找出第index个样本的所有数据
        gt_path = os.path.join(self.gt_path, self.gt_list[index])
        for im_path in img_list:
            image_path = os.path.join(self.image_path, im_path)
            image_iter = read_ENVI(image_path)
            if self.image is None:
                self.image = image_iter
                ndvi = cal_ndvi(self.image)  # 计算指数
                ndwi = cal_ndwi(self.image)
                self.image = mean_std_normalize(self.image, self.im_cmean, self.im_cstd)  #先对RGB进行归一化
                self.image = np.concatenate((self.image, np.expand_dims(ndvi, axis=2)), axis=2)  # 合并指数和光谱
                self.image = np.concatenate((self.image, np.expand_dims(ndwi, axis=2)), axis=2)
            else:
                image_iter = image_iter[:, :, :13]
                if self.sentinel_use_flag:
                    self.image = np.concatenate((self.image, image_iter), axis=2)
        self.mask = read_ENVI(gt_path)
        if self.data_enhancement_flag:
            image_enhancement = self.trans(image=self.image, mask=self.mask)
            self.image = image_enhancement['image']
            self.mask = image_enhancement['mask']

        # 训练数据预处理，拆分成minibatch
        if self.train_flag:
            positive_train_indicator = self.get_positive_train_indicator()
            unlabeled_train_indicator = self.get_unlabeled_train_indicator()
            # 1*7*W*H
            blob = divisible_pad([np.concatenate([self.image.transpose(2, 0, 1),
                                                  self.mask[None, :, :],
                                                  positive_train_indicator[None, :, :],
                                                  unlabeled_train_indicator[None, :, :]], axis=0)], 16)

            self.im = blob[0, :self.image.shape[-1], :, :]
            self.positive_train_indicator = blob[0, -2, :, :]
            self.unlabeled_train_indicator = blob[0, -1, :, :]

            self.positive_inds_list, self.unlabeled_inds_list = minibatch_sample(self.positive_train_indicator,
                                                                                 self.unlabeled_train_indicator,
                                                                                 sub_num_iter=self.sub_num_iter,
                                                                                 seed=self.seeds_for_minibatchsample.pop())
            self.actual_batch_size = len(self.positive_inds_list)
        else:
            print('not belong to training data, because train_flag was false')

    # def resample_minibatch(self):
    #     if self.train_flag:
    #         self.positive_inds_list, self.unlabeled_inds_list = minibatch_sample(self.positive_train_indicator,
    #                                                                              self.unlabeled_train_indicator,
    #                                                                              sub_num_iter=self.sub_num_iter,
    #                                                                              seed=self.seeds_for_minibatchsample.pop())

    def get_positive_train_indicator(self):
        # 返回一个type==cls的index map
        gt_mask_flatten = self.mask.ravel()
        positive_train_indicator = np.zeros_like(gt_mask_flatten)
        positive_train_indicator[np.where(gt_mask_flatten == self.cls)[0]] = 1
        positive_train_indicator = positive_train_indicator.reshape(self.mask.shape)
        return positive_train_indicator

    def get_unlabeled_train_indicator(self):
        # 随机生成num_train_samples_per_class * ratio 数量的index
        rs = np.random.RandomState(self._seed)
        gt_mask_flatten = self.mask.ravel()
        unlabeled_train_indicator = np.zeros_like(gt_mask_flatten)
        # unlabeled_train_indicator[:4000] = 1
        unlabeled_train_indicator[
        :min(int(self.num_train_samples_per_class * self.ratio), gt_mask_flatten.shape[0])] = 1
        rs.shuffle(unlabeled_train_indicator)
        unlabeled_train_indicator = unlabeled_train_indicator.reshape(self.mask.shape)
        return unlabeled_train_indicator

    def read_tif_list(self, file_path, img_type='*.tif'):
        img_list = fnmatch.filter(os.listdir(file_path), img_type)  # tif还是TIF注意区别
        return img_list

    def __getitem__(self, idx):
        # print('training', idx)
        self.preset(idx)
        return self.im, np.stack(self.positive_inds_list, axis=0), np.stack(self.unlabeled_inds_list, axis=0)

    def __len__(self):
        return int(len(self.image_list) / 4)


class TestDataset(data.Dataset):  # 单张训练影像的
    def __init__(self, config, data_root_path, train_flag=False):
        super(TestDataset, self).__init__()
        ##########图像路径及文件名##########
        self.data_root_path = data_root_path
        self.train_flag = train_flag  # 是否训练模式
        self.sentinel_use_flag = sentinel_use_flag
        self.init_user(config)

    def init_user(self, config):
        if self.train_flag:
            keywords = 'train'
        else:
            keywords = 'test'
        self.cls = config['dataset'][keywords]['ccls']
        self.sub_num_iter = config['dataset'][keywords]['sub_minibatch']
        self.num_train_samples_per_class = config['dataset'][keywords]['num_positive_train_samples']
        self.ratio = config['dataset'][keywords]['ratio']
        self.im_cmean = config['dataset'][keywords]['im_cmean']
        self.im_cstd = config['dataset'][keywords]['im_cstd']
        self._seed = 2333  # 随机数种子
        self._rs = np.random.RandomState(self._seed)
        # set list length = 9999 to make sure seeds enough
        self.seeds_for_minibatchsample = [e for e in self._rs.randint(low=2147483648, size=9999)]
        # 读取目标文件下的所有tif文件
        self.image = None
        self.image_path = os.path.join(self.data_root_path, keywords, 'Image')
        self.gt_path = os.path.join(self.data_root_path, keywords, 'Label')
        self.image_list = self.read_tif_list(self.image_path)
        self.gt_list = self.read_tif_list(self.gt_path)
        self.actual_batch_size = 0  # 实际的batch size，由于样本数无法被指定的bs整除产生的变化

    def preset(self, index):
        self.image = None  # 清空上一次image
        index = max(0, index)  # 防止左侧索引溢出
        index = min(index, int(len(self.image_list) / 4) - 1)  # 防止右侧索引溢出
        img_list = fnmatch.filter(self.image_list, str(index) + '*')  # 找出第index个样本的所有数据
        gt_path = os.path.join(self.gt_path, self.gt_list[index])
        for im_path in img_list:
            image_path = os.path.join(self.image_path, im_path)
            image_iter = read_ENVI(image_path)
            if self.image is None:
                self.image = image_iter
                ndvi = cal_ndvi(self.image)  # 计算指数
                ndwi = cal_ndwi(self.image)
                self.image = mean_std_normalize(self.image, self.im_cmean, self.im_cstd)
                self.image = np.concatenate((self.image, np.expand_dims(ndvi, axis=2)), axis=2)  # 合并指数和光谱
                self.image = np.concatenate((self.image, np.expand_dims(ndwi, axis=2)), axis=2)
            else:
                image_iter = image_iter[:, :, :13]
                if self.sentinel_use_flag:
                    self.image = np.concatenate((self.image, image_iter), axis=2)
        self.mask = read_ENVI(gt_path)
        self.positive_test_mask = self.get_positive_train_indicator()
        self.negative_test_mask = np.negative(self.get_positive_train_indicator())

    def get_positive_train_indicator(self):
        # 返回一个type==cls的index map
        gt_mask_flatten = self.mask.ravel()
        positive_train_indicator = np.zeros_like(gt_mask_flatten)
        positive_train_indicator[np.where(gt_mask_flatten == self.cls)[0]] = 1
        positive_train_indicator = positive_train_indicator.reshape(self.mask.shape)
        return positive_train_indicator

    def read_tif_list(self, file_path, img_type='*.tif'):
        img_list = fnmatch.filter(os.listdir(file_path), img_type)  # tif还是TIF注意区别
        return img_list

    def __getitem__(self, idx):
        self.preset(idx)
        return self.image.transpose((2, 0, 1)).astype(np.float32), self.mask

    def __len__(self):
        return int(len(self.image_list) / 4)


class TestDataset4HS(data.Dataset):  # 高分辨率验验证集
    def __init__(self, config, data_root_path, shuffle=False):
        super(TestDataset4HS, self).__init__()
        ##########图像路径及文件名##########
        self.data_root_path = data_root_path
        self.sentinel_use_flag = False
        self.stuff = shuffle
        self.cls = config['dataset']['train']['ccls']
        self.im_cmean = config['dataset']['train']['im_cmean']
        self.im_cstd = config['dataset']['train']['im_cstd']
        # 读取目标文件下的所有tif文件
        self.image = None
        self.image_path = os.path.join(self.data_root_path, 'Image')
        self.gt_path = os.path.join(self.data_root_path, 'Label')
        self.image_list = self.read_tif_list(self.image_path)
        self.gt_list = self.read_tif_list(self.gt_path)

    def preset(self, index):
        self.image = None  # 清空上一次image
        index = max(0, index)  # 防止左侧索引溢出
        index = min(index, int(len(self.image_list)) - 1)  # 防止右侧索引溢出
        img_list = fnmatch.filter(self.image_list, "CaIrr" + str(index + 1) + '*')  # 找出第index个样本的所有数据
        gt_path = os.path.join(self.gt_path, self.gt_list[index])
        for im_path in img_list:
            image_path = os.path.join(self.image_path, im_path)
            image_iter = read_ENVI(image_path)
            if self.image is None:
                self.image = image_iter
                ndvi = cal_ndvi(self.image)  # 计算指数
                ndwi = cal_ndwi(self.image)
                self.image = mean_std_normalize(self.image, self.im_cmean, self.im_cstd)
                self.image = np.concatenate((self.image, np.expand_dims(ndvi, axis=2)), axis=2)  # 合并指数和光谱
                self.image = np.concatenate((self.image, np.expand_dims(ndwi, axis=2)), axis=2)
            else:
                image_iter = image_iter[:, :, :13]
                if self.sentinel_use_flag:
                    self.image = np.concatenate((self.image, image_iter), axis=2)
        self.mask = read_ENVI(gt_path)

    def read_tif_list(self, file_path, img_type='*.tif'):
        img_list = fnmatch.filter(os.listdir(file_path), img_type)  # tif还是TIF注意区别
        return img_list

    def __getitem__(self, idx):
        self.preset(idx)
        return self.image.transpose((2, 0, 1)).astype(np.float32), self.mask

    def __len__(self):
        if self.sentinel_use_flag:
            return int(len(self.image_list) / 4)
        else:
            return int(len(self.image_list))
