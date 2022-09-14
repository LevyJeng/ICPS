import torch

from torch import nn

from torch.nn import functional as F

# FC层在keras中叫做Dense层，正在pytorch中交Linear层

X = torch.randn(10, 5)
Y = torch.mm(X, torch.FloatTensor([[1], [2], [3], [4], [5]]))  # 看成就是矩阵乘法
# torch.matmul()的用法比torch.mm更高级，torch.mm只适用于二维矩阵，而torch.matmul可以适用于高维。当然，对于二维的效果等同于torch.mm（）。
print(Y)

# Example1
l1 = nn.Linear(5, 4)  # 其实就是矩阵乘法
l2 = nn.Linear(4, 3)
net = nn.Sequential(l1, l2, nn.ReLU(), nn.Linear(3, 2))
nest_net = nn.Sequential(net, net, nn.ReLU())  # 可以嵌套sequence
print("net[0]:")
print(net[0])
print("net:")
print(net)
print("X:")
print(X)
print("net(X):")
print(net(X))


# Example2
# 自定义一个多层网络，其实自定一个单层也是这种写法，没什么本质的区别
class MLP(nn.Module):  # 继承nn.Module
    def __init__(self, *args):
        super().__init__()
        for block in args:
            self._modules[block] = block

    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X


mlp = MLP(nn.Linear(5, 10), nn.Linear(10, 3))
Y = mlp(X)  # 得到mlp对象后就可以直接处理输入的X，应该是可以帮你自动调用forward函数
print(Y)

# Example3
print("state_dict可以获取网络层参数：")
print(net.state_dict())  # relu不会显示出来，一开始的权重参数不是0，而是会进行kaiming初始化，为了更好地训练
print("也可以通过直接访问参数来获取权重和偏置")
print(net[0].weight)
print(net[0].bias)  # requires_grad=True,说明在训练过程中是需要进行梯度下降优化的
print("还可以通过下面的方式进行访问：")
print(*[(name, param.shape) for name, param in net.named_parameters()])


# Example4
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)  # 手动来初始化m的参数
        # nn.init.xavier_normal(m.weight)
        nn.init.zeros_(m.bias)
        # nn.init.constant_(m.bias, 2)


net.apply(init_normal)  # 会遍历整个神经网络进行处理，xavier、kaiming也是一种初始化
print("net:")
print(net.state_dict())


# Example5
class MyLayer(nn.Module):  # 继承nn.Module,实现一个单层
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


layer = MyLayer()
Y = layer(torch.FloatTensor([1, 2, 3]))

# Example6
torch.save(X, "X-file")
torch.save([X, X], "list-file")
torch.save({'X': X}, "dict-file")

torch.save(mlp.state_dict(), "state-file")
copy = MLP()
copy.eval()  # 复原

copy.load_state_dict(torch.load("state-file"))
X = torch.load("X-file")
