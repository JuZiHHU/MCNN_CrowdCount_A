# 统一导入packages
import glob
import os
import re

import cv2
import matplotlib
import  h5py
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

torch.backends.cudnn.benchmark = False # 因为输入图片大小不确定，所以设置其为 False，下文有详细介绍
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 使用 gpu 进行训练

'''
数据集展示
读取数据集中的某个样本，如下图所示。为了方便自由修改查看，将路径拆分为多个部分
'''
sample_idx = 6 # 样本编号
sample_root = "datasets/part_A_final/train_data/" # 样本根目录
img_path = os.path.join(sample_root, f"images/IMG_{sample_idx}.jpg") # 图片
gt_mat_path = os.path.join(sample_root, f"ground_truth/GT_IMG_{sample_idx}.mat") # 标签
'''
可视化图片如下图所示
gt 中包含该图片的所有需要的信息，image_info 包含有图片的每个人头的二维坐标，以及图片中人头的数目，
将其在图中标注出来，结果如下图所示。
'''
img = plt.imread(img_path)
gt = io.loadmat(gt_mat_path)
plt.imshow(img);plt.show()
print(gt)

xys = gt["image_info"][0][0][0][0][0] # 获取图像中所有的人头位置
xs = [i[0] for i in xys]; ys = [i[1] for i in xys] # 分别获取横坐标和纵坐标，方便展示
plt.imshow(img); plt.plot(xs,ys,'rx'); plt.show()

'''
密度图制作
MCNN 是基于密度图来进行人群计数的，所以我们需要根据 .mat 文件制作训练所需的密度图
'''
def make_density(img, points):
    img_shape = [img.shape[0],img.shape[1]] # 获取图像形状
    print(f"shape of imgs: {img_shape} , number of gaussian kernels: {len(points)}",end="\t")
    density = np.zeros(img_shape, dtype=np.float32) # 初始化一个全 0 矩阵
    gt_count = len(points)
    if gt_count == 0: # 如果图像中没有人，则返回全 0 矩阵
        return density

    leafsize = 2048
    # 构建 kdtree
    tree = spatial.KDTree(points.copy(), leafsize=leafsize)
    distances, locations = tree.query(points, k=4) # 在这里选取 k 为4

    print('processing...', end='\t')
    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1]) < img_shape[0] and int(pt[0]) < img_shape[1]:
            pt2d[int(pt[1]), int(pt[0])] = 1.
        else:
            continue
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2.
        density += scipy.ndimage.filters.gaussian_filter(
            pt2d, sigma, mode='constant')
    print('done.')
    return density
'''
调用函数来制作密度图（part_A），注意这里输出只保留了前十项。
'''

# 根目录
root = 'datasets/'

# 总的路径列表
path_sets = []
# 添加路径列表（part_A）
path_sets.append(os.path.join(root, 'part_A_final/train_data', 'images'))
path_sets.append(os.path.join(root, 'part_A_final/test_data', 'images'))
# 添加路径列表（part_B）
path_sets.append(os.path.join(root,'part_B_final/train_data','images'))
path_sets.append(os.path.join(root,'part_B_final/test_data','images'))

# 保存路径
for save_root in path_sets:
    save_root = save_root.replace('datasets','temp').replace('images', 'ground_truth')
    if not os.path.exists(save_root):
        os.makedirs(save_root)

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')): # 返回所有匹配的文件路径列表
        img_paths.append(img_path)

for idx,img_path in enumerate(img_paths):
    save_path = img_path.replace('.jpg', '.npy').replace('images', 'ground_truth').replace('datasets','temp')
    if os.path.exists(save_path):
        continue
    print(img_path, end='\t')
    mat = io.loadmat(img_path.replace('.jpg', '.mat').replace(
        'images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
    img = plt.imread(img_path)
    k = np.zeros((img.shape[0], img.shape[1]))
    points = mat["image_info"][0, 0][0, 0][0]
    k = make_density(img, points)
    np.save(save_path, k)
#    if idx>2:break #控制时间，可放开

density_idx = 138 # 样本编号
density_root = "./temp/part_A_final/train_data/" # 样本根目录
density_path = os.path.join(density_root, f"ground_truth/IMG_{density_idx}.npy") # 图片
density = np.load(density_path)
plt.imshow(density); plt.show()