# PROJECT_NAME: learn_rnn
# DATE: 2023/7/21
# USER: du_jl
# DESCRIPTION:训练词向量

import os, torch
import numpy as np
from gensim.models.word2vec import Word2Vec
import pickle
def split_poetry(file="poetry_7.txt"):
    # open 打开txt文件 read进行读取
    org_data = open(file=file, mode="r", encoding='utf-8').read()
    # " ".join() 用空格进行连接字符串
    split_data = " ".join(org_data)

    # 打开文件 f作为指针指向该文件
    with open(file="split_7.txt", mode="w+", encoding="utf-8") as f:
        # 向该文件中写入数据(字符串)
        f.write(split_data)


# 训练词向量,参数：原始文件，切割文件
def train_word_2_vector(org_file="poetry_7.txt", split_file="split_7.txt"):

    params_file = "params.pkl"

    # 不存在切割文件
    if os.path.exists(split_file) == False:
        split_poetry(org_file)

    # 打开文件，读取文件，切分文件,形成list
    split_data = open(split_file, "r", encoding="utf-8").read().split('\n')
    all_data = open(org_file, "r", encoding="utf-8").read().split('\n')

    # 存在params_file(已经训练过词向量)
    if os.path.exists(params_file):
        return all_data, pickle.load(open(params_file, "rb"))


    # 实例化Word2Vec，参数：数据；训练成的向量维数，最小出现次数（即最小出现几次，如果最小出现1次，就是每个字都要进行训练，如果最小出现3次，那么3次以内可以不进行训练）；进程数
    model = Word2Vec(split_data, vector_size=107, min_count=1,  workers=6)

    # pickle要会用 -- 把训练好的东西(debug查看)存进一个文件，方便使用
    # all_data -- 原始数据以'\n'切割后形成的list
    # model.syn1neg -- w1(n*m)n个不重复的字，每个字vector_size维去表示
    # index_to_key -- index_2_vector
    # key_to_index -- vector_2_index
    pickle.dump((model.syn1neg, model.wv.index_to_key, model.wv.key_to_index), open("params.pkl","wb"))

    return all_data, (model.syn1neg, model.wv.index_to_key, model.wv.key_to_index)


 # 训练词向量
    # all_data -- 以‘\n'分割的原始数据; w1 -- （5364，107）词向量矩阵; index_2_vec -- 索引2向量; vec_2_index -- 向量2索引
    all_data, (w1, index_2_vec, vec_2_index) = train_word_2_vector()

# 训练词向量
# all_data -- 以‘\n'分割的原始数据; w1 -- （5364，107）词向量矩阵; index_2_vec -- 索引2向量; vec_2_index -- 向量2索引
# all_data, (w1, index_2_vec, vec_2_index) = train_word_2_vector()