import torchvision
from torch import nn, reshape
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

CVdataset = torchvision.datasets.CIFAR10("..\\dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
CVLoader = DataLoader(CVdataset, batch_size=64)


class Pumbaa(nn.Module):
    def __init__(self):
        super(Pumbaa, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

    # print(imgs.shape)
    # print(output

# Convolution Layer
pumbaaConv = Pumbaa()
step = 0
writer = SummaryWriter("../CVLogs")
for data in CVLoader:
    imgs, targets = data
    output = pumbaaConv(imgs)
    writer.add_images("input", imgs, step)
    output = reshape(output, (-1, 1, 30, 30))
    writer.add_images("Conv2d", output, step)
    step = step + 1
writer.close()
