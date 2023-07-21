# PROJECT_NAME: learn_rnn
# DATE: 2023/7/21
# USER: du_jl
# DESCRIPTION: 生成藏头诗
import torch
import numpy as np
from train_word_2_vector import train_word_2_vector


def generate_head():
    # 生成的古诗
    results = ""
    # 头
    word1, word2, word3, word4 = input("请输入四个字：")
    # 索引
    word1_index = vec_2_index[word1]
    word2_index = vec_2_index[word2]
    word3_index = vec_2_index[word3]
    word4_index = vec_2_index[word4]

    # 得到第一句
    results += word1
    # （layer_num, batch_num, dihhen_num)
    h_0 = torch.zeros(2, 1, hidden_num)
    c_0 = torch.zeros(2, 1, hidden_num)
    for i in range(6):
        word_embeding = torch.tensor(w1[word1_index].reshape(1,1,-1))
        pre, (h_0, c_0) = model(word_embeding, h_0, c_0)
        word_index = int(torch.argmax(pre))
        results += index_2_vec[word_index]
    results += '，'
    # 得到第二句
    results += word2
    h_0 = torch.zeros(2, 1, hidden_num)
    c_0 = torch.zeros(2, 1, hidden_num)
    for i in range(6):
        word_embeding = torch.tensor(w1[word2_index].reshape(1,1,-1))
        pre, (h_0, c_0) = model(word_embeding, h_0, c_0)
        word_index = int(torch.argmax(pre))
        results += index_2_vec[word_index]
    results += '。'
    # 得到第三句
    h_0 = torch.zeros(2, 1, hidden_num)
    c_0 = torch.zeros(2, 1, hidden_num)
    results += word3
    for i in range(6):
        word_embeding = torch.tensor(w1[word3_index].reshape(1, 1, -1))
        pre, (h_0, c_0) = model(word_embeding, h_0, c_0)
        word_index = int(torch.argmax(pre))
        results += index_2_vec[word_index]
    results += '，'
    # 得到第四句
    results += word4
    h_0 = torch.zeros(2, 1, hidden_num)
    c_0 = torch.zeros(2, 1, hidden_num)
    for i in range(6):
        word_embeding = torch.tensor(w1[word4_index].reshape(1, 1, -1))
        pre, (h_0, c_0) = model(word_embeding, h_0, c_0)
        word_index = int(torch.argmax(pre))
        results += index_2_vec[word_index]
    results += '。'
    print(results)



model = torch.load("lstm_(99, 99).pth")
# print(model)
hidden_num = 256    # 和train保持一致
all_data, (w1, index_2_vec, vec_2_index) = train_word_2_vector() # 训练词向量
generate_head() # 生成藏头诗


