# 统一导入packages
import glob
import os
import re

import cv2
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import scipy
import scipy.ndimage
import scipy.io as io
import scipy.spatial as spatial
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchsummary import summary
from tensorboardX import SummaryWriter

'''
构造一个 dataset 类，继承于 torch.utils.data.Dataset，
我们只需要重写他的初始化函数 __init__，返回数据集长度的函数 __len__，以及如何读取每一个样本的 __getitem__

__len__ ：较为简单，只需调用 len() 来获取数据集长度即可
__init__ : 定义一些基本参数，如图片的根目录等，需要与 __getitem__ 一起配合完成数据的读取
__getitem__ : 接受 int 类型的参数 index，即想要读取的样本的索引，根据索引返回该样本。
'''

class CrowdDataset(Dataset):
    def __init__(self, img_root, gt_dmap_root, gt_downsample=1):
        '''
        img_root: 图片的根目录.
        gt_dmap_root: 真实密度图的根目录.
        gt_downsample: 默认为0，表示模型的输出与输入图像大小相同.
        '''
        self.img_root = img_root
        self.gt_dmap_root = gt_dmap_root
        self.gt_downsample = gt_downsample

        self.img_names = [filename for filename in os.listdir(img_root)
                          if os.path.isfile(os.path.join(img_root, filename))] # 获取所有图片的文件名
        self.n_samples = len(self.img_names) # 数据集的长度

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_name = self.img_names[index]
        img = plt.imread(os.path.join(self.img_root, img_name))
        if len(img.shape) == 2:  # 将单通道灰度图扩展为三通道
            img = img[:, :, np.newaxis]
            img = np.concatenate((img, img, img), 2)
        gt_dmap = np.load(os.path.join(self.gt_dmap_root, img_name.replace('.jpg', '.npy')))
        # 对图像和密度图进行下采样
        if self.gt_downsample > 1:
            ds_rows = int(img.shape[0]//self.gt_downsample)
            ds_cols = int(img.shape[1]//self.gt_downsample)
            img = cv2.resize(img, (ds_cols*self.gt_downsample,
                                   ds_rows*self.gt_downsample))
            # 顺序转换为 (channel,rows,cols)
            img = img.transpose((2, 0, 1))
            gt_dmap = cv2.resize(gt_dmap, (ds_cols, ds_rows))
            gt_dmap = gt_dmap[np.newaxis, :, :] * \
                self.gt_downsample*self.gt_downsample

            img_tensor = torch.tensor(img, dtype=torch.float)
            gt_dmap_tensor = torch.tensor(gt_dmap, dtype=torch.float)

        return img_tensor, gt_dmap_tensor

