# encoding:utf-8
import torch
import numpy as np
import matplotlib.pyplot as plt  # 导入作图相关的包
from torch import nn


# 定义RNN模型
class Rnn(nn.Module):
    def __init__(self, INPUT_SIZE):
        super(Rnn, self).__init__()

        # 定义RNN网络,输入单个数字.隐藏层size为[feature, hidden_size]
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True  # 注意这里用了batch_first=True 所以输入形状为[batch_size, time_step, feature]
        )
        # 定义一个全连接层,本质上是令RNN网络得以输出
        self.out = nn.Linear(32, 1)

    # 定义前向传播函数
    def forward(self, x, h_state):
        # 给定一个序列x,每个x.size=[batch_size, feature].同时给定一个h_state初始状态,RNN网络输出结果并同时给出隐藏层输出
        r_out, h_state = self.rnn(x, h_state)
        outs = []
        for time in range(r_out.size(1)):  # r_out.size=[1,10,32]即将一个长度为10的序列的每个元素都映射到隐藏层上.
            outs.append(self.out(r_out[:, time, :]))  # 依次抽取序列中每个单词,将之通过全连接层并输出.r_out[:, 0, :].size()=[1,32] -> [1,1]
        return torch.stack(outs, dim=1), h_state  # stack函数在dim=1上叠加:10*[1,1] -> [1,10,1] 同时h_state已经被更新


TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.02

model = Rnn(INPUT_SIZE)
print(model)

loss_func = nn.MSELoss()  # 使用均方误差函数
optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # 使用Adam算法来优化Rnn的参数,包括一个nn.RNN层和nn.Linear层

h_state = None  # 初始化h_state为None

for step in range(300):
    # 人工生成输入和输出,输入x.size=[1,10,1],输出y.size=[1,10,1]
    start, end = step * np.pi, (step + 1) * np.pi

    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)

    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    # 将x通过网络,长度为10的序列通过网络得到最终隐藏层状态h_state和长度为10的输出prediction:[1,10,1]
    prediction, h_state = model(x, h_state)
    h_state = h_state.data  # 这一步只取了h_state.data.因为h_state包含.data和.grad 舍弃了梯度
    # 反向传播
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()

    # 优化网络参数具体应指W_xh, W_hh, b_h.以及W_hq, b_q
    optimizer.step()

# 对最后一次的结果作图查看网络的预测效果
plt.plot(steps, y_np.flatten(), 'r-')
plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
plt.show()
