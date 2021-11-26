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
from dataloder import  CrowdDataset
from  mode import  MCNN

torch.backends.cudnn.benchmark = True # 因为输入图片大小不确定，所以设置其为 False，下文有详细介绍
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 使用 gpu 进行训练

'''
根据前面定义好的类 CrowdDataset 来构造一个 dataloader，
训练时通过迭代 dataloader 来获取数据。我们可以通过dataloader 来实现一些基础功能，如将数据集打乱（shuffle=True），
设置batch_size，多线程加载数据（num_workers）等。
需要注意的是，由于数据集中图片大小不一样，所以设置 batch_size=1，
同样的原因，在上文中，我们需要设置 torch.backends.cudnn.benchmark=False（详细原因可参考此文章）。
'''

# 训练集
img_root = './datasets/part_B_final/train_data/images'
gt_dmap_root = './temp/part_B_final/train_data/ground_truth'
dataset = CrowdDataset(img_root, gt_dmap_root, 4)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
# 测试集
test_img_root = './datasets/part_B_final/test_data/images'
test_gt_dmap_root = './temp/part_B_final/test_data/ground_truth'
test_dataset = CrowdDataset(test_img_root, test_gt_dmap_root, 4)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)


print(device)
mcnn = MCNN().to(device) # 模型实例化

'''
在训练模型时，每训练完一个 epoch 测试一下模型效果如何，并将模型的参数保存下来，防止训练被意外中断以及方便测试。
整个训练过程中，不断对比每个 epoch 的结果，将最好的结果记录下来。
'''
criterion = nn.MSELoss(reduction='sum').to(device) # 损失函数
optimizer = torch.optim.SGD(mcnn.parameters(), lr=1e-6, momentum=0.95) # 优化器
summary(mcnn,(3,1000,500))

if not os.path.exists('./temp/models_B'):
    os.makedirs('./temp/models_B')

'''
与训练集不同，测试集部分，为了与论文保持一致，我们测试的是指标 MAE
'''

min_mae, min_epoc = 1000, 0 # 最小的 MAE 和其对应的 epoch
train_loss_list, epoch_list, test_error_list = [], [], [] # 方便可视化
writer = SummaryWriter('./temp/logs/') # 可视化训练过程
num_epochs = 1030#1000
mcnn.to(device)

cp = "./temp/models_B/epoch_999.param"
mcnn.load_state_dict(torch.load(cp))
start_epoch = 1000#这里是因为之前已经训练了999个模型，继续上次的运行，初次运行可以注释这三行代码



for epoch in range(start_epoch,num_epochs):
    mcnn.train()
    epoch_loss=0
    for i,(img,gt_dmap) in enumerate(dataloader):
        current_batch = epoch * len(dataloader) + i
        img=img.to(device)
        gt_dmap=gt_dmap.to(device)
        # 前向传播
        et_dmap=mcnn(img)
        # 计算损失函数
        loss=criterion(et_dmap,gt_dmap)
        epoch_loss+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('train/batch_mse_loss', loss.item(), current_batch)
        #print(i)
        #if i>5:break
    epoch_list.append(epoch)
    train_loss_list.append(epoch_loss/len(dataloader))
    torch.save(mcnn.state_dict(),'./temp/models/epoch_'+str(epoch)+".param")
    writer.add_scalar('train/epoch_mse_loss', epoch_loss/len(dataloader), epoch)

    mcnn.eval()
    mae=0
    for i,(img,gt_dmap) in enumerate(test_dataloader):
        img=img.to(device)
        gt_dmap=gt_dmap.to(device)
        # 前向传播
        et_dmap=mcnn(img)
        # 计算 MAE
        mae+=abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
        del img,gt_dmap,et_dmap
        #print(i)
        #if i>5:break
    if mae/len(test_dataloader)<min_mae:
        min_mae=mae/len(test_dataloader)
        min_epoch=epoch
    test_error_list.append(mae/len(test_dataloader))
    writer.add_scalar('test/epoch_mae_loss', mae/len(test_dataloader), epoch)
    print(f"epoch:{epoch} | loss:{epoch_loss/len(dataloader)} | error:{mae/len(test_dataloader)} | min_mae:{min_mae} | min_epoch:{min_epoch}")
    plt.plot(epoch_list,train_loss_list); plt.title("MSE during the training process");plt.show()
    plt.plot(epoch_list,test_error_list); plt.title("MAE during the testing process");plt.show()

'''
模型评估
'''

mae_list, e_list = [], []
cp_list = glob.glob("./temp/models_B/*.param") # optionA: 所有 checkpoints
#cp_list = [f"temp/models/epoch_{idx}.param" for idx in [50, 100, 150, 200, 250, 300, 400, 500, 600]] # optionB: 部分 checkpoints
# cp_list = [f"temp/models/epoch_0.param"]
for cp in cp_list:
    epoch = int(re.findall(r"\d+", cp)[0])
    mcnn.load_state_dict(torch.load(cp))
    mcnn.eval()
    mae=0
    for i,(img,gt_dmap) in enumerate(test_dataloader):
        img=img.to(device)
        gt_dmap=gt_dmap.to(device)
        # forward propagation
        et_dmap=mcnn(img)
        mae+=abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
        #print(i)
        #if i>1:break
    mean_mae=mae/len(test_dataloader)
    mae_list.append(mean_mae)
    e_list.append(epoch)
    print(f"current epoch: {epoch} | mae: {mean_mae}")

plt.plot(e_list,mae_list)
plt.scatter(e_list,mae_list)
best_mae = min(mae_list)
best_epoch = e_list[mae_list.index(best_mae)]
plt.show()
print(f"best epoch in the selected epochs: {best_epoch} | best mae: {best_mae}")

