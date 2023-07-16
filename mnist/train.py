# PROJECT_NAME: learn_rnn
# DATE: 2023/7/11
# USER: du_jl
# DESCRIPTION:实战案例：MNIST手写字体识别



import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from module import *

# 1.判断是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2.可视化
writer = SummaryWriter(log_dir="../log")

# 3.定义超参数
learning_rate = 0.01 # 学习率
BATCH_SIZE = 128 # 每批读取的数据大小
EPOCHS = 30 # 训练10轮

# 4.下载mnist数据集, 并创建数据集的可迭代对象，也就说一个batch一个batch的读取
trainsets = datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
testsets = datasets.MNIST(root="./data",train=False, transform= transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=trainsets, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=testsets, batch_size=BATCH_SIZE, shuffle=True)

train_lenth = len(trainsets)
test_lenth = len(testsets)

# 查看类别/标签
# class_names = trainsets.classes
# print(class_names)
# 查看一批baatch的数据
# images, labels = next(iter(test_loader))
# print(images.shape)
# writer = SummaryWriter("log")
# writer.add_images("train", images)
# writer.close()

# 5.实例化RNN模型
# model = RNN_Model()
# model.to(device)

# 初始化
input_dim = 28 #图片的维度
hidden_dim = 100 # 自己设置
layer_dim = 2 # 2层RNN
output_dim = 10 # 10种图片
model = RNN_Model(input_dim=input_dim, hidden_dim=hidden_dim, layer_dim=layer_dim, output_dim=output_dim)
model.to(device)
print(model)

# 6.定义损失函数
criterion = nn.CrossEntropyLoss()

# 7.定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 8.参数
total_train_step = 0 # 训练步数
total_test_step = 0 # 测试步数
sequence_dim = 28 # 序列长度

# 9.开始训练
for epoch in range(EPOCHS):
    #训练
    model.train()
    print("----第{}次训练开始----".format(epoch))
    for data in train_loader:
        imgs, targets = data
        imgs = imgs.view(-1, sequence_dim, input_dim).to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        # 优化器
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数{}，损失{}".format(total_train_step,loss))
            writer.add_scalar("train", loss, total_train_step)


    #测试
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            imgs = imgs.view(-1, sequence_dim, input_dim).to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print("整体测试集上的loss:{}".format(total_test_loss))
    print("整体测试集上的正确率accuracy:{}:".format(total_accuracy/test_lenth))
    writer.add_scalar("test", total_test_loss, total_test_step)
    writer.add_scalar("accuracy", total_accuracy/test_lenth, total_test_step)
    total_test_step += 1


    # 保存模型
    torch.save(model, "rnn_{}.pth".format(epoch))
    print("模型已保存！")


writer.close()