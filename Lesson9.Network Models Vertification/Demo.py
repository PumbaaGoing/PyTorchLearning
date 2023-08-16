# @File : Demo.py
# @Time : 2023-08-03 11:07
# @Author : Pumbaa

import torch
import torchvision
from torch import nn
from PIL import Image
from torch.nn import MaxPool2d, Conv2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

# imgs_class = Image.open("../CVImages/CIF10Dataset_Classes.jpg")
# imgs_class.show()

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

# testModel = torch.load("..\\Lesson7.Pre-trained Models\\testModel.pth")
# print(testModel)

pumbaaNN = torch.load("../Lesson8.Network Models  Exercise/Saved Models/pumbaaNN_gpu37.pth")
# pumbaaNN = torch.load("..\\Lesson7.Pre-trained Models\\testModel.pth")
# print(pumbaaNN)

transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                             torchvision.transforms.ToTensor()])

image_path = "../CVImages/Auto3.jpg"
img = Image.open(image_path)
# img.show()
# print(img)


img = transforms(img)
img = torch.reshape(img, (1, 3, 32, 32))
img = img.cuda()
# print(img.shape)

with torch.no_grad():
    output = pumbaaNN(img)

print(output.argmax(1))



