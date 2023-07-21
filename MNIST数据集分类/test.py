# PROJECT_NAME: learn_rnn
# DATE: 2023/7/15
# USER: du_jl
# DESCRIPTION:测试
import PIL
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("./testlog")

# 1.读取图片
img_path = "img/img_11.png"
img = Image.open(img_path)
# 2.反转图像颜色
img = PIL.ImageOps.invert(img)
print(img)

# 3.转换成单通道 mode:'L'单通道灰度图   ‘RGB'三通道
img = img.convert('L')
print(img)

# 4.修改PIL img的尺寸 + 转换成tensor
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((28,28)),
                                            torchvision.transforms.ToTensor()])
img = transform(img)
print(img.shape)
writer.add_image("test", img, 1)

# 修改tensor img的尺寸
# img = torch.reshape(img, (1, 28, 28))
# imgs = img.view(-1, 28, 28)
# print(img.shape)


# 上gpu
img = img.cuda()

# 4.测试
module = torch.load("rnn_29.pth")
print(module)
module.eval()
with torch.no_grad():
    output = module(img)
print(output)
print(output.argmax(1))

writer.close()
