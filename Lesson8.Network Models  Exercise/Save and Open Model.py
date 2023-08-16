# @File : Save and Open Model.py
# @Time : 2023-08-02 19:42
# @Author : Pumbaa

# save new model & data --- Method1
import torch
import torchvision
vgg_tosave = torchvision.models.vgg16()
# torch.save(vgg_tosave, "vgg_savetest1.pth")

# save only new data by dictionary --- Method2[Recommended]
torch.save(vgg_tosave.state_dict(), "vgg_savetest2.pth" )


# load new model --- Method1
# vgg_open1 = torch.load("vgg_savetest1.pth")
# print(vgg_open1)

# load new data to origin model --- Method2[Recommended]
# load model's data
# vgg_open2_data = torch.load("vgg_savetest2.pth")
# print(vgg_open2_data)

# load data to the new model
vgg_open2 = torchvision.models.vgg16()
# vgg_open2.load_state_dict(vgg_open2_data)
print(vgg_open2)