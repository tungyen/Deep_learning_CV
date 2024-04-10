from torch.utils.data import Dataset
import torch
from torchvision import transforms
import random
import os
from PIL import Image as im

flower_dict = {'daisy':0, 'dandelion':1, 'roses':2, 'sunflowers':3, 'tulips':4}

data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


class flowerDataset(Dataset):
    def __init__(self, dataPath, train=True):
        self.dataPath = dataPath
        self.train = train
        self.flowerDict = flower_dict
        
        if self.train:
            self.transform = data_transform['train']
            datasetPath = os.path.join(self.dataPath, 'train')
        else:
            self.transform = data_transform['val']
            datasetPath = os.path.join(self.dataPath, 'val')
        
        flower_class = os.listdir(datasetPath) 
        self.datasetPath = datasetPath
        
        flowerFiles = []
        labels = []
        
        for cla in flower_class:
            class_path = os.path.join(self.datasetPath, cla)
            class_files = os.listdir(class_path)
            
            for class_file in class_files:
                flowerFiles.append(class_file)
                labels.append(cla)

        data = list(zip(flowerFiles, labels))
        random.shuffle(data)
        flowerFiles, labels = zip(*data)
        self.flowerFiles = list(flowerFiles)
        self.labels = list(labels)
                
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        imgPath = os.path.join(self.datasetPath, self.labels[idx], self.flowerFiles[idx])
        img = im.open(imgPath)
        img = self.transform(img)
            
        return img, torch.tensor(flower_dict[self.labels[idx]], dtype=torch.long)
    

# if __name__ == '__main__':
#     data = flowerDataset('..\\Dataset\\flowerDataset\\flower_data')
#     img, label = data[5]
#     print(img.shape)
    