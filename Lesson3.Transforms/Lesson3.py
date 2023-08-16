from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

imgPath = "data/hymenopteraData/train/ants_image/0013035.jpg"
img = Image.open(imgPath)

writer = SummaryWriter("logs")


tensorTrans = transforms.ToTensor()
tensorImg = tensorTrans(img)

writer.add_image("TensorImg", tensorImg)

writer.close()
