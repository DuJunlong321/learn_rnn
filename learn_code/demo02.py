# PROJECT_NAME: learn_rnn
# DATE: 2023/7/11
# USER: du_jl
# DESCRIPTION: rnn：参数维度，输入特征维度，输出特征维度

import torch
from torch import nn

# demo1:1层
# input_size     100维的单词
# hidden_size    隐藏层的维度，即：隐藏层里面有几个节点，把特征降维或升维
# num_layers    层数，默认1
rnn = nn.RNN(input_size=100, hidden_size=10, num_layers=1)

# 各层的参数：目前只设置了一层
print(rnn._parameters.keys())

# weight_hh_l0  第0层隐藏层的参数，w_hh表示：举个例子w_ax，第二个下标意味着w_ax要乘以某个𝑥类型的量，然后第一个下标𝑎表示它是用来计算某个𝑎类型的变量。
# torch.Size([10, 10]) torch.Size([10, 100])
# w_hh:在cell里不需要进行维度转换
# w_ih:在cell里需要进行维度转换，将输入的100维的特征转换成10维
print(rnn.weight_hh_l0.shape,rnn.weight_ih_l0.shape)


# demo2:这里层数设置成2
rnn2 = nn.RNN(input_size=100, hidden_size=10, num_layers=2)

# 各层的参数
print(rnn2._parameters.keys())

# 查看各层的参数维度
# torch.Size([10, 10]) torch.Size([10, 100])
# w_hh_l0:在cell里不需要进行维度转换
# w_ih_l0:在cell里需要进行维度转换，将输入的100维的特征转换成10维
print(rnn2.weight_hh_l0.shape,rnn2.weight_ih_l0.shape)

# torch.Size([10, 10]) torch.Size([10, 10])
# 经过第一层的转换，现已经都是10维的，不再需要转换
print(rnn2.weight_hh_l1.shape,rnn2.weight_ih_l1.shape)


# demo3:这里层数设置成4,且不向rnn里传递初始h0
rnn3 = nn.RNN(input_size=100, hidden_size=20, num_layers=4)
print(rnn3._parameters.keys())

# 输入特征:
# 第一个参数：10个时间t,或者说10个单词，就是时间上从左往右进行10次cell
# 第二个参数：batches,或者说3句话
# 第三个参数：每个单词用100维的向量表示（例如100维的one_hot)
# 总的来说就是：3句话，每句话有10个单词，每个单词用100维的向量表示
x = torch.randn(10, 3, 100)

# out,h的区别在于out是上面出来的所有输出，h是右面出来的所有输出，见img.png
out, h = rnn3(x)

# out.shape:torch.Size([10, 3, 20])
# out:[时间t,batches,隐藏层维度]，即：[t, batches, hidden_size]
# 表示：一共t个时间状态（t个单词），batches是送进来几组数据（几句话），hidden_size表示被cell降成几维了
print(out.shape)

# h.shape:torch.Size([4, 3, 20])
# h:[num_layers,batches,hidden_size]
# 表示：一共4层，batches是送进来的几组数据，hidden_size表示被cell降成几维了
print(h.shape)