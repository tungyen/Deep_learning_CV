from torch.utils.data import Dataset
import torch
from PIL import Image as im
import os
import json

class flowerDataset(Dataset):
    def __init__(self, root, transform=None):
        
        flowerClasses = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
        flowerClasses.sort()
        classIndex = dict((key, value) for value, key in enumerate(flowerClasses))
        self.classIndex = classIndex
        jsonStr = json.dumps(dict((val, key) for key, val in classIndex.items()), indent=4)
        with open('classIndex.json', 'w') as jsonFile:
            jsonFile.write(jsonStr)
        
        self.root = root
        self.transform = transform
        
        self.img_paths = []
        self.labels = []
        
        for cla in os.listdir(root):
            for path in os.listdir(os.path.join(root, cla)):
                self.img_paths.append(os.path.join(root, cla, path))
                self.labels.append(classIndex[cla])
                
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = im.open(self.img_paths[idx])
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
