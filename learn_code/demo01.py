# PROJECT_NAME: learn_rnn
# DATE: 2023/7/11
# USER: du_jl
# DESCRIPTION:引入

import torch

batch_size = 1
seq_len= 3
input_size = 4
hidden_size = 2
num_layers = 1

rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size,
                   num_layers=num_layers)


inputs = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(num_layers, batch_size, hidden_size)

out, hidden = rnn(inputs, hidden)

print('Output size:', out.shape)
print('Output:', out)
print('Hidden size:', hidden.shape)
print('Hidden:', hidden)

