import torch.nn as nn
import torch
from torch.autograd import Variable

class ADSHLoss(nn.Module):
    def __init__(self, gamma, code_length, num_train):
        super(ADSHLoss, self).__init__()
        self.gamma = gamma
        self.code_length = code_length
        self.num_train = num_train

    def forward(self, u, V, S, V_omega):
        batch_size = u.size(0)
        # 0是行数，1是列数
        V = Variable(torch.from_numpy(V).type(torch.FloatTensor).cuda())
        # from_numpy Numpy桥，将numpy.ndarray 转换为pytorch的 Tensor。 返回的张量tensor和numpy的ndarray共享同一内存空间。
        #修改一个会导致另外一个也被修改。返回的张量不能改变大小。
        V_omega = Variable(torch.from_numpy(V_omega).type(torch.FloatTensor).cuda())
        S = Variable(S.cuda())
        #会记录当前选择的GPU，并且分配的所有CUDA张量将在上面创建。可以使用torch.cuda.device上下文管理器更改所选设备。
        square_loss = (u.mm(V.t())-self.code_length * S) ** 2
        #mm 矩阵相乘
        quantization_loss = self.gamma * (V_omega - u) ** 2
        loss = (square_loss.sum() + quantization_loss.sum()) / (self.num_train * batch_size)
        return loss
