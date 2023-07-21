# PROJECT_NAME: learn_rnn
# DATE: 2023/7/21
# USER: du_jl
# DESCRIPTION:自定义dataset

import numpy as np

class Mydataset:
    # 初始化数据
    # 加载数据
    def __init__(self, all_data, w1, index_2_vec, vec_2_index):
        self.all_data = all_data
        self.w1 = w1
        self.index_2_vec = index_2_vec
        self.vec_2_index = vec_2_index

    # 取一条数据
    def __getitem__(self, index):
        # 一首古诗的字
        a_poetry_words = self.all_data[index]
        # 一首古诗的索引
        a_poetry_index = [self.vec_2_index[word] for word in a_poetry_words]

        # 求x,y: x--训练数据 y--训练标记
        # 这首古诗作为训练数据，它的index，为了求它的特征
        xs_index = a_poetry_index[:-1]
        # 这首古诗的特征 (用到了python的特性，eg:[5，3，8，9]意为取5，3，8，9行)
        # 一首古诗的特征是：(31,107) 31个字，每个字107维，在外面还会进行batches,形成xs_embeddings(batches, 31, 107)
        xs_embedding = self.w1[xs_index]

        # 训练这首古诗的标记，它的index。 这里不用求y的特征，最后求损失的时候，torch不需要特征，只需要索引
        ys_index = a_poetry_index[1:]

        # 这样的话， ys_index是一个list,每个元素是一个tensor(5),list无法在touch上跑
        # return xs_embedding, ys_index

        # 这样ys_index就是一个tensor(31,5),可以在torch上跑
        return xs_embedding, np.array(ys_index).astype(np.int64)

    # 获取数据集的长度
    def __len__(self):
        return len(self.all_data)
