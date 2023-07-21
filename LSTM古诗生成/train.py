# PROJECT_NAME: learn_rnn
# DATE: 2023/7/18
# USER: du_jl
# DESCRIPTION:

# 数据处理
import numpy as np
# 数据处理，数据封装
from torch.utils.data import Dataset,DataLoader
# 模型
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from model import *
from train_word_2_vector import *
from dataset import *

#自动生成古诗
def generate_poetry_auto():
    # 生成的古诗
    result = ""
    # 在[0,words_num)产生随机数
    word_index = np.random.randint(0, words_num, 1)[0]
    result += index_2_vec[word_index]

    # (layer_num, batch_num, dihhen_num)
    h_0 = torch.zeros(2, 1, hidden_num)
    c_0 = torch.zeros(2, 1, hidden_num)
    for i in range(31):
        word_embedding = torch.tensor(w1[word_index].reshape(1, 1, -1))\
        # STLM 返回三个值
        # out -- 预测值；   h -- h_0;    c -- c_0
        pre, (h_0, c_0) = model(word_embedding, h_0, c_0)
        # 挑选概率最大预测值
        word_index = int(torch.argmax(pre))
        result += index_2_vec[word_index]
    print(result)

if __name__=="__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer = SummaryWriter("logs")

    # -------------------------------超参数---------------------------------------------
    batch_size = 32
    hidden_num = 256
    epochs = 100
    lr = 0.005

    # ------------------------------训练词向量--------------------------------------------
    # all_data -- 以‘\n'分割的原始数据;
    # w1 -- （5364，107）词向量矩阵;
    # index_2_vec -- 索引2向量;
    # vec_2_index -- 向量2索引
    all_data, (w1, index_2_vec, vec_2_index) = train_word_2_vector()
    # -------------------------------创建数据---------------------------------------------
    # 创建数据实例
    # 一个batch一个batch的加载数据
    dataset = Mydataset(all_data, w1, index_2_vec, vec_2_index)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    # --------------------------------  参数 ------------------------------------------------
    # 获取不重复的字个数， 训练的向量维数
    words_num, embedding_num = w1.shape
    total_step = 0
    step = 0

    # ------------------------------- 创建模型-----------------------------------------------

    model = Mymodel(words_num=words_num, embedding_num=embedding_num, hidden_num=hidden_num)
    model = model.to(device)
    # print(model)
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 损失
    cross_entropy_loss = nn.CrossEntropyLoss()

    #  ----------------------------------  训练  --------------------------------------------------
    for epoch in range(epochs):
        # dataloader 范围是：[0, 197) 即:batch_size的范围。 因为：总数据/batch_num = 6291/32 = 197
        # 也就是说会有197个batch
        for batch_index, (xs_embedding, ys_index) in enumerate(dataloader):

            xs_embedding = xs_embedding.to(device)
            ys_index = ys_index.to(device)
            pre, _ = model(xs_embedding)
            loss = cross_entropy_loss(pre, ys_index.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # step += 1
            # print("total_step:{}, step:{}, batche_index:{}".format(total_step, step, batch_index))
            if batch_index % 100 == 0:
                print("total_step：{}，batch:{}, 损失：{}".format(total_step, batch_index, loss))
                writer.add_scalar("loss", loss,  global_step=total_step)
                generate_poetry_auto()
            total_step += 1
        # 注意：保存的是模型实例model，而非类：Mymodel
        torch.save(model, "lstm_{}.pth".format(epoch))

    writer.close()