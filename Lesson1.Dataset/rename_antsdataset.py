import os

rootDir = "data/hymenopteraData/train"
targetDir = "ants_image"
imgPath = os.listdir(os.path.join(rootDir, targetDir))
label = targetDir.split('_')[0]
outDir = "ants_label"
for i in imgPath:
    fileName = i.split('.jpg')[0]
    with open(os.path.join(rootDir,outDir,"{}.txt".format(fileName)),'w') as f:
        f.write(label)