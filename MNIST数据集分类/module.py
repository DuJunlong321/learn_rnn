# PROJECT_NAME: learn_rnn
# DATE: 2023/7/15
# USER: du_jl
# DESCRIPTION:RNN模型

import torch.nn
from torch import nn


class RNN_Model(nn.RNN):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNN_Model, self).__init__(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        #   (layer_dim, batch_dim, hidden_dim)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_size).requires_grad_().to('cuda')
        # 分离隐层状态，防止梯度爆炸
        out, hn = self.rnn(x, h0.detach())
        out = self.fc(out[:, -1,:])
        return out

