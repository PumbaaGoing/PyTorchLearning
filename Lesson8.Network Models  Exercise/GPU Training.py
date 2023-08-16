# @File : GPU Training.py
# @Time : 2023-08-03 10:20
# @Author : Pumbaa
import time

import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# define device
device = torch.device("cuda")

# from Model import *  # import everything from Model.py
from torch.nn import MaxPool2d, Conv2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

# import dataset

train_data = torchvision.datasets.CIFAR10("..\\dataset", train=True, transform=torchvision.transforms.ToTensor()
                                          , download=True)
test_data = torchvision.datasets.CIFAR10("..\\dataset", train=False, transform=torchvision.transforms.ToTensor()
                                         , download=True)
# check size
train_data_size = len(train_data)
test_data_size = len(test_data)
print("lenth of train_data is:{}".format(train_data_size))
print("lenth of test_data is:{}".format(test_data_size))

# load dataset
train_loader = DataLoader(train_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)

# build a new model
# class PumbaaNN(nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super(PumbaaNN, self).__init__()
#         self.module = Sequential(
#             Conv2d(3, 32, 5, padding=2),
#             MaxPool2d(2),
#             Conv2d(32, 32, 5, padding=2),
#             MaxPool2d(2),
#             Conv2d(32, 64, 5, padding=2),
#             MaxPool2d(2),
#             Flatten(),
#             Linear(1024, 64),
#             Linear(64, 10)
#         )
#     def forward(self, x):
#         x = self.module(x)
#         return x

# pumbaaNN = PumbaaNN()
pumbaaNN = torchvision.models.vgg16()
pumbaaNN.add_module('add_linear', nn.Linear(1000, 10))
print(pumbaaNN)

if torch.cuda.is_available():
    pumbaaNN = pumbaaNN.cuda()

# load Loss Function
loss_fun = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    loss_fun = loss_fun.cuda()

# load Optimizer
learning_rate = 0.01
optimizer = torch.optim.SGD(pumbaaNN.parameters(), lr=learning_rate)


# To record parameters related to NN
# train steps
total_train_steps = 0
# test steps
total_test_steps = 0
# training epoch
epoch = 50

# load Tensorboard
writer = SummaryWriter("..\\CVLogs")
start_time = time.time()
# Begin Training

# to some certain Models
pumbaaNN.train()
for i in range(epoch):
    print("Current Epoch:{}".format(i + 1))
    for data in train_loader:
        imgs, targets = data

        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()

        outputs = pumbaaNN(imgs)
        loss = loss_fun(outputs, targets)

        # Optimizer work
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_steps = total_train_steps + 1
        if total_train_steps % 100 == 0:
            print("Training steps: {}, loss: {}".format(total_train_steps, loss))
            writer.add_scalar("train_loss", loss, total_train_steps)

    # Test in test dataset
    # to some certain Models
    pumbaaNN.eval()
    total_test_loss = 0
    total_accurate_outcomes = 0
    with torch.no_grad():  # Disable the optimizer to optimize the Model when testing/evaluating
        for data in test_loader:
            imgs, targets = data
            # open GPUs
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()

            outputs = pumbaaNN(imgs)

            loss = loss_fun(outputs, targets)
            total_test_loss = total_test_loss + loss

            accurate_outcomes = (outputs.argmax(1) == targets).sum()
            total_accurate_outcomes = total_accurate_outcomes + accurate_outcomes
            accuracy = total_accurate_outcomes / test_data_size

    writer.add_scalar("test_loss", total_test_loss, total_test_steps)

    print("Epoch {} Model Accuracy is {}, J_test is {}".format((i + 1), accuracy, total_test_loss/test_data_size))
    writer.add_scalar("accuracy", accuracy, total_test_steps)

    total_test_steps = total_test_steps + 1
    # Save the Model of current epoch
    torch.save(pumbaaNN, "Saved Models\\pumbaaNN_gpu{}.pth".format(i + 1))
    print("Epoch {} Model has been saved successfully".format(i + 1))

writer.close()
end_time = time.time()
print("Done. Total time is {}s".format(end_time - start_time))

