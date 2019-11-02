# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 20:38:45 2019

@author: beach
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import numpy as np
import net_kaiming

from visdom import Visdom

# (Hyper parameters)
batch_size = 64
learning_rate = 1e-2
num_epochs = 20

if __name__ == '__main__':
    data_tf = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    train_dataset = datasets.MNIST(
        root=r'C:\Users\beach\Desktop\Study\深度学习技术与应用\作业1\data', train=True, transform=data_tf, download=True)
    test_dataset = datasets.MNIST(root=r'C:\Users\beach\Desktop\Study\深度学习技术与应用\作业1\data', train=False, transform=data_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #batch批量梯度下降
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = net_kaiming.SimpleNet(28 * 28, 300, 100, 10)
    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.CrossEntropyLoss() #损失函数
    optimizer = optim.SGD(model.parameters(), lr=learning_rate) #随机梯度下降的优化器

    epoch = 0
    x,y = 0,0
    viz = Visdom(env='my_wind1')
    win = viz.line(
            X=np.array([x]),
            Y=np.array([y]),
            opts=dict(showlegend=True))
    for data in train_loader:
        img, label = data
        img = img.view(img.size(0), -1) #batch*28*28
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        out = model(img)
        loss = criterion(out, label)
        print_loss = loss.data.item()

        optimizer.zero_grad() #梯度清0
        loss.backward() #求到
        optimizer.step() #更新
        epoch+=1
        
        x=epoch
        y=print_loss
        viz.line(
                X=np.array([x]),
                Y=np.array([y]),
                win=win,
                update='append')
        
        if epoch%100==0:
            print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))

    model.eval() # batch 测试不做BatchNormalization
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:
        img, label = data
        img = img.view(img.size(0), -1)
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()

        out = model(img)
        loss = criterion(out, label)
        eval_loss+=loss.data.item()*label.size(0)
        _, pred = torch.max(out, 1) # 1是维数，第二维
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
        eval_loss / (len(test_dataset)),
        eval_acc / (len(test_dataset))
    ))
