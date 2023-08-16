# @File : Lesson7.py
# @Time : 2023-08-02 11:46
# @Author : Pumbaa
import numpy as np
import torch

import torchvision
from torch import nn
train_data = torchvision.datasets.CIFAR10("..\\dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)
np.set_printoptions(suppress=True)
# train_data = torchvision.datasets.ImageNet("..\\dataset", split="train", download=True, transform=torchvision.transforms.ToTensor())

testModel = torchvision.models.vgg16()

# Change Model
# testModel.add_module('add_linear', nn.Linear(1000, 10))
testModel.classifier[6] = nn.Linear(4096, 10)
print(testModel)

torch.save(testModel, "testModel.pth")
