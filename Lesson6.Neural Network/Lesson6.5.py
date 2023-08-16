# Loss Function & Backward Network & Optimizer

import torchvision
from torch import nn, reshape
from torch.nn import Conv2d, MaxPool2d, ReLU, Sigmoid, Sequential, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Loss Function

CVdataset = torchvision.datasets.CIFAR10("..\\dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
CVLoader = DataLoader(CVdataset, batch_size=64)


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
loss = nn.CrossEntropyLoss()
# print(pumbaaNN)
# step = 0
# writer = SummaryWriter("../CVLogs")
for data in CVLoader:
    imgs, targets = data
    outputs = pumbaaNN(imgs)
    result_loss = loss(outputs, targets)
    result_loss.backward()
    print("ok")
    # print(result_loss)
    # print(outputs.shape)
    # writer.add_images("input", imgs, step)
    # writer.add_images("Nonlinear", output, step)
    # step = step + 1
# writer.close()
