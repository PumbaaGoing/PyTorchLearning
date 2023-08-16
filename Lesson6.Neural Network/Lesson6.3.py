import torchvision
from torch import nn, reshape
from torch.nn import Conv2d, MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# PoolLayers

CVdataset = torchvision.datasets.CIFAR10("..\\dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
CVLoader = DataLoader(CVdataset, batch_size=64)


class Pumbaa(nn.Module):
    def __init__(self):
        super(Pumbaa, self).__init__()
        self.conv2 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        Output = self.conv2(input)
        return Output

    # print(imgs.shape)
    # print(output

# Convolution Layer
Maxpool = Pumbaa()
step = 0
writer = SummaryWriter("../CVLogs")
for data in CVLoader:
    imgs, targets = data
    output = Maxpool(imgs)
    writer.add_images("input", imgs, step)
    writer.add_images("Maxpool", output, step)
    step = step + 1
writer.close()
