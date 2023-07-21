# PROJECT_NAME: learn_rnn
# DATE: 2023/7/21
# USER: du_jl
# DESCRIPTION: lstm模型

from torch import nn
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


class Mymodel(nn.Module):
    def __init__(self, embedding_num, words_num, hidden_num=64, num_layers=2):

        super().__init__()

        self.embedding_num = embedding_num
        self.hidden_num = hidden_num
        self.words_num = words_num

        # input_size 特征维度，即词向量训练后的特征维度
        # hidden_size 隐层维度，即隐层有几个圈
        self.lstm = nn.LSTM(input_size=embedding_num, hidden_size=hidden_num, num_layers=num_layers, batch_first=True,proj_size = 0)
        # 随机失活
        self.dropout = nn.Dropout(0.2)
        # 将[0,1]层拉平
        self.flatten = nn.Flatten(0, 1)
        # 线性层(全连接),做分类,实质是看最后在所有不重合字里分成什么字
        # in_features 隐层维度
        # out_features  所有不重合的字
        self.linear = nn.Linear(in_features=hidden_num, out_features=words_num)
        # self.cross_entropy_loss = nn.CrossEntropyLoss();
    def forward(self, xs_embedding, h_0=None, c_0=None):
        xs_embedding = xs_embedding.to(device)
        if h_0 == None or c_0 == None:
            # h_0 (layer_num,batches_num,hidden_num)
            # h_0 = torch.tensor(np.zeros((3 * 1, xs_embedding.shape[0], self.hidden_num), dtype=np.float32))
            h_0 = torch.zeros(2, xs_embedding.shape[0], self.hidden_num)
            c_0 = torch.tensor(np.zeros((2 * 1, xs_embedding.shape[0], self.hidden_num), dtype=np.float32))
        h_0 = h_0.to(device)
        c_0 = c_0.to(device)
        # (5，3，64)
        hidden, (h_0, c_0) = self.lstm(xs_embedding, (h_0, c_0))
        # (5,3,64)
        hidden_dropout = self.dropout(hidden)
        # (155,64)
        dropout_flatten = self.flatten(hidden_dropout)
        # (155,3542)
        pre = self.linear(dropout_flatten)
        return pre, (h_0, c_0)
