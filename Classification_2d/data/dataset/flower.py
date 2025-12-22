from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image as im
import os
import json

class FlowerDataset(Dataset):
    def __init__(self, data_path, split, transforms=None):
        assert split == "train" or split == "val" or split == "test"

        data_path = os.path.join(data_path, split)
        flower_classes = [cla for cla in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, cla))]
        flower_classes.sort()
        class_index = dict((key, value) for value, key in enumerate(flower_classes))
        self.class_index = class_index
        json_str = json.dumps(dict((val, key) for key, val in class_index.items()), indent=4)
        with open('classIndex.json', 'w') as json_file:
            json_file.write(json_str)
            
        json_path = 'classIndex.json'
        with open(json_path, "r") as f:
            self.class_dict = json.load(f)
        
        self.data_path = data_path
        self.transforms = transforms
        
        self.img_paths = []
        self.labels = []
        
        for cla in os.listdir(data_path):
            for path in os.listdir(os.path.join(data_path, cla)):
                self.img_paths.append(os.path.join(data_path, cla, path))
                self.labels.append(class_index[cla])
                
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = im.open(self.img_paths[idx])
        label = self.labels[idx]
        if self.transforms is not None:
            img, label = self.transforms(img, label)
            
        return img, label

    def get_class_dict(self):
        return self.class_dict
    
    @staticmethod
    def collate_fn(batch):
        imgs, labels = tuple(zip(*batch))
        imgs = torch.stack(imgs, dim=0)
        labels = torch.as_tensor(labels)
        return imgs, labels