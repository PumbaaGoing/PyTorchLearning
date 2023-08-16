from torch.utils.data import Dataset
from PIL import Image
import os


class OpenData(Dataset):

    def __init__(self, rootDir, labelDir):
        self.rootDir = rootDir
        self.labelDir = labelDir
        self.path = os.path.join(self.rootDir, self.labelDir)
        self.imgPath = os.listdir(self.path)

    def __getitem__(self, index):
        imageName = self.imgPath[index]
        imgItemPath = os.path.join(self.rootDir, self.labelDir, imageName)
        img = Image.open(imgItemPath)
        label = self.labelDir
        return img, label

    def __len__(self):
        return len(self.imgPath)


rootDir = "data/hymenopteraData/train"
antsLabelDir = "ants"
beesLabelDir = "bees"
antsDataset = OpenData(rootDir, antsLabelDir)
beesDataset = OpenData(rootDir, beesLabelDir)

trainDataset = antsDataset + beesDataset