import torch
import torchvision
from torch import nn, reshape
from torch.nn import Conv2d, MaxPool2d, ReLU, Sigmoid, Dropout2d, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Dropout Function

CVdataset = torchvision.datasets.CIFAR10("..\\dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
CVLoader = DataLoader(CVdataset, batch_size=64)


class Pumbaa(nn.Module):
    def __init__(self):
        super(Pumbaa, self).__init__()
        self.dropout = Dropout2d(0.5)

    def forward(self, input):
        Output = self.dropout(input)
        return Output

    # print(imgs.shape)
    # print(Output)


dropout2d = Pumbaa()
step = 0
writer = SummaryWriter("../CVLogs")
for data in CVLoader:
    imgs, targets = data
    output = dropout2d(imgs)
    writer.add_images("input1", imgs, step)
    writer.add_images("Dropout", output, step)
    # print(imgs.shape)
    # print(output.shape)
    step = step + 1
writer.close()


# Linear Function
class Pumbaa2(nn.Module):

    def __init__(self):
        super(Pumbaa2, self).__init__()
        self.linear = Linear(196608, 10)

    def forward(self, input):
        Output = self.linear(input)
        return Output


linear = Pumbaa2()
step = 0
writer = SummaryWriter("../CVLogs")
for data in CVLoader:
    imgs, targets = data
    # temp = torch.flatten(imgs)
    # print(temp.shape)
    output = linear(imgs)
    writer.add_images("input2", imgs, step)
    writer.add_images("Linear", output, step)
    step = step + 1
writer.close()
