from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
imagePath = "data/hymenopteraData/val/bees/10870992_eebeeb3a12.jpg"
imgPIL = Image.open(imagePath)
imgArray = np.array(imgPIL)

writer.add_image("bees", imgArray, 1, dataformats='HWC')
# y =2x
for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)

writer.close()
