from torch.utils.data import Dataset
import torch
from PIL import Image as im

import os
import sys
import json
import random


def dataLoading(rootPath, valRatio=0.2):
    random.seed(0)
    flowerClasses = [cla for cla in os.listdir(rootPath) if os.path.isdir(os.path.join(rootPath, cla))]
    flowerClasses.sort()
    classIndex = dict((key, value) for value, key in enumerate(flowerClasses))
    jsonStr = json.dumps(dict((val, key) for key, val in classIndex.items()), indent=4)
    with open('classIndex.json', 'w') as jsonFile:
        jsonFile.write(jsonStr)
        
    supported = [".jpg", ".JPG", ".png", ".PNG"]
    trainImgPaths = []
    trainLabels = []
    valImgPaths = []
    valLabels = []
    classNum = []
    
    for cla in flowerClasses:
        claPath = os.path.join(rootPath, cla)
        imgPaths = [os.path.join(rootPath, cla, i) for i in os.listdir(claPath) if os.path.splitext(i)[-1] in supported]
        imgPaths.sort()
        claIndex = classIndex[cla]
        classNum.append(len(imgPaths))
        valPaths = random.sample(imgPaths, k=int(len(imgPaths) * valRatio))
        
        for imgPath in imgPaths:
            if imgPath in valPaths:
                valImgPaths.append(imgPath)
                valLabels.append(claIndex)
            else:
                trainImgPaths.append(imgPath)
                trainLabels.append(claIndex)
                
    return trainImgPaths, trainLabels, valImgPaths, valLabels


class flowerDataset(Dataset):
    def __init__(self, imgPath, labels, transform=None):
        self.imgPath = imgPath
        self.labels = labels
        self.transform = transform
                   
    def __len__(self):
        return len(self.imgPath)
        
    def __getitem__(self, idx):
        img = im.open(self.imgPath[idx])
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
            
        return img, label
    
    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels