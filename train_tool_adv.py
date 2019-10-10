# -*- coding: utf-8 -*-
__author__ = "kk"



sequence_length = 28  # 序列长度，将图像的每一列作为一个序列
input_size = 4  # 输入数据的维度
hidden_size = 128  # 隐藏层的size
num_layers = 2  # 有多少层

num_classes = 2
batch_size = 1
num_epochs = 20
learning_rate = 0.1

global x_simple
global y_label

import torch

class EncoderRNNWithVector(torch.nn.Module):
    def __init__(self, input_size, hidden_size, out_size, n_layers=1, batch_size=1):
        super(EncoderRNNWithVector, self).__init__()

        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.out_size = out_size

        # 这里指定了 BATCH FIRST
        self.gru = torch.nn.GRU(input_size, hidden_size, n_layers, batch_first=True)
       # 加了一个线性层，全连接
        self.out = torch.nn.Linear(hidden_size, out_size)

    def forward(self, word_inputs, hidden):
        # -1 是在其他确定的情况下，PyTorch 能够自动推断出来，view 函数就是在数据不变的情况下重新整理数据维度
        # batch, time_seq, input
        inputs = word_inputs.view(self.batch_size, -1, self.hidden_size)
       # hidden 就是上下文输出，output 就是 RNN 输出
        output, hidden = self.gru(inputs, hidden)
        output = self.out(output)

        # 仅仅获取 time seq 维度中的最后一个向量
        # the last of time_seq
        output = output[:,-1,:]

        return output, hidden

    def init_hidden(self):
        # 这个函数写在这里，有一定迷惑性，这个不是模型的一部分，是每次第一个向量没有上下文，在这里捞一个上下文，仅此而已。
        hidden = torch.autograd.Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        return hidden


import random
# 这里随机生成一个 Tensor，维度是 1000 x 10 x 200；其实就是1000个句子，每个句子里面有10个词向量，每个词向量 200 维度，其中的值符合 NORMAL 分布。
_xs = torch.tensor(x_simple) # torch.randn(10, 4, 1)
_ys = y_label # []


# 标签值 0 - 5 闭区间
# for i in range(1000):
#     _ys.append(random.randint(0, 5))

# 隐层 200，输出 6，隐层用词向量的宽度，输出用标签的值得个数 （one-hot)
encoder_test = EncoderRNNWithVector(4, 4, 2)

criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(encoder_test.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(encoder_test.parameters(), lr=0.01 )

for i in range(_xs.size()[0]):
    encoder_hidden = encoder_test.init_hidden()
    input_data = torch.autograd.Variable(_xs[i])
    output_labels = torch.autograd.Variable(torch.LongTensor([_ys[i]]))
    #print(output_labels)

    encoder_outputs, encoder_hidden = encoder_test(input_data, encoder_hidden)

    optimizer.zero_grad()
    loss = criterion(encoder_outputs, output_labels)
    loss.backward()
    optimizer.step()

    print("loss: ", loss.data)

# Save the Model
# torch.save(rnn.state_dict(), 'rnn.pkl')