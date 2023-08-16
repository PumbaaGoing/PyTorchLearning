# @File : Model.py
# @Time : 2023-08-02 20:12
# @Author : Pumbaa

import torch
import torchvision
from torch import nn

from torch.nn import MaxPool2d, Conv2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

# build network model
class PumbaaNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(PumbaaNN, self).__init__()
        self.module = Sequential(
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
    def forward(self, x):
        x = self.module(x)
        return x

# test model
if __name__ == '__main__':
    pumbaaNN = PumbaaNN()
    input = torch.ones((64, 3, 32, 32))
    output = pumbaaNN(input)
    print(output.shape)