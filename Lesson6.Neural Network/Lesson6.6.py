# @File : Lesson6.6.py
# @Time : 2023-08-01 20:29
# @Author : Pumbaa

# Test NN Settings
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class PumbaaNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(PumbaaNN, self).__init__(*args, **kwargs)
        # self.conv1 = Conv2d(3, 32, 5, padding=2)
        # self.maxpool1 = MaxPool2d(2)
        # self.conv2 = Conv2d(32, 32, 5, padding=2)
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(32, 64, 5, padding=2)
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten = Flatten()
        # self.linear1 = Linear(1024, 64)
        # self.linear2 = Linear(64, 10)
        self.module1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )


    def forward(self, input):
       # input = self.conv1(input)
       # input = self.maxpool1(input)
       # input = self.conv2(input)
       # input = self.maxpool2(input)
       # input = self.conv3(input)
       # input = self.maxpool3(input)
       # input = self.flatten(input)
       # input = self.linear1(input)
       # input = self.linear2(input)
       input = self.module1(input)
       return input


pumbaaNN = PumbaaNN()
# print(pumbaaNN)
input = torch.ones((64, 3, 32, 32))
output = pumbaaNN(input)
# print(output.shape)
writer = SummaryWriter("../CVLogs")
writer.add_graph(pumbaaNN, input)
writer.close()