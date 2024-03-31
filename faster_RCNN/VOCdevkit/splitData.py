import os
import random

filePath = "./VOC2012/Annotations"
if not os.path.exists(filePath):
    print("This folder does not exist!")
    exit(1)
    
valRatio = 0.5

fileNames = sorted([file.split(".")[0] for file in os.listdir(filePath)])
fileNum = len(fileNames)
valIndex = random.sample(range(0, fileNum), k=int(fileNum * valRatio))
trainFiles = []
valFiles = []
for ind, fileName in enumerate(fileNames):
    if ind in valIndex:
        valFiles.append(fileName)
    else:
        trainFiles.append(fileName)
        
try:
    train_f = open("train.txt", "x")
    val_f = open("val.txt", "x")
    train_f.write("\n".join(trainFiles))
    val_f.write("\n".join(valFiles))
except FileExistsError as e:
    print(e)
    exit(1)
        