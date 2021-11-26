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
from mode import MCNN
torch.backends.cudnn.benchmark = False # 因为输入图片大小不确定，所以设置其为 False，下文有详细介绍
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 使用 gpu 进行训练


def plot_tensor(tensor, show=True):
    ret = tensor.squeeze(0).squeeze(0).cpu().detach().numpy() # 去除两个维度，转化为 numpy 类型
    if show:
        plt.imshow(ret)
        plt.show()
    return ret

best_epoch = 951 #560 # 由训练过程得到
best_cp = f"temp/models_B/epoch_{best_epoch}.param"
mcnn = MCNN().to(device) # 模型实例化
mcnn.to("cpu").load_state_dict(torch.load(best_cp))
mcnn = mcnn.eval()


# 读取指定图片（原图片，密度图和 ground-truth）
idx =106
img_path = f"datasets/part_B_final/test_data/images/IMG_{idx}.jpg"
gt_path = f"temp/part_B_final/test_data/ground_truth/IMG_{idx}.npy"
gtmat_path = f"datasets/part_B_final/test_data/ground_truth/GT_IMG_{idx}.mat"
img = plt.imread(img_path)
img_tensor = torch.tensor(img.transpose((2, 0, 1)),dtype=torch.float).unsqueeze(0).to("cpu") # 转化为tensor类型
gt = np.load(gt_path)
output = mcnn(img_tensor)
output_number = output.sum() # 求和以获得模型检测得到的人群总数
gt_number = io.loadmat(gtmat_path)["image_info"][0][0][0][0][1][0][0] # 读取ground-truth中的人群数目

plt.figure(figsize=(12,4))
plt.subplot(131);plt.imshow(img)
plt.subplot(132);plt.imshow(gt)
plt.subplot(133);plot_tensor(output)

print(f"输出结果：{output_number.int()}，真实结果：{gt_number}")

save_root = "temp/results"
if not os.path.exists(save_root):
    os.makedirs(save_root)
save_name = img_path.split("/")[-1]
save_path = os.path.join(save_root,save_name)
plt.imsave(save_path, plot_tensor(output, show=False)) # 保存输出图片

with open(save_path.replace(".jpg",".txt"),"w") as f: # 保存txt结果文件
    f.write(f"img:{save_path}, people:{output_number.int()}")


'''


idx = 17 # 1,2,3,4
img_path = f"pictures/test{idx}.jpg"
img = plt.imread(img_path)
img_tensor = torch.tensor(img.transpose((2, 0, 1)),dtype=torch.float).unsqueeze(0).to("cpu")
output = mcnn(img_tensor) # 模型测试输出
output_number = output.sum() # 人群数目检测结果

plt.figure(figsize=(8,4))
plt.subplot(121);plt.imshow(img)
plt.subplot(122);plot_tensor(output)
plt.show()

print(f"输出结果：{output_number.int()}")
save_root = "result/"
save_name = img_path.split("/")[-1]
save_path = os.path.join(save_root,save_name)
plt.imsave(save_path, plot_tensor(output, show=False)) # 保存图片

with open(save_path.replace(".jpg",".txt"),"w") as f:
    f.write(f"img:{save_path}, people:{output_number.int()}") #保存结果为txt文件

'''