import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from PIL import Image
import transforms as T


train_dirs = ["jena", "zurich", "weimar", "ulm", "tubingen", "stuttgart",
              "strasbourg", "monchengladbach", "krefeld", "hanover",
              "hamburg", "erfurt", "dusseldorf", "darmstadt", "cologne",
              "bremen", "bochum", "aachen"]
val_dirs = ["frankfurt", "munster", "lindau"]
test_dirs = ["berlin", "bielefeld", "bonn", "leverkusen", "mainz", "munich"]
loc_list = [train_dirs, val_dirs, test_dirs]

mode_to_loc = {"train": 0, "val": 1, "test": 2}

class CityScapes(Dataset):
    def __init__(self, root='./Dataset/cityscapes', mode='train', transform=None):
        self.img_path = os.path.join(root, 'leftImg8bit_trainvaltest', 'leftImg8bit', mode)
        self.label_path = os.path.join(root, 'gtFine_trainvaltest', 'gtFine', mode)
        
        self.img_h = 1024
        self.img_w = 2048

        self.new_img_h = 512
        self.new_img_w = 1024
        self.transform = transform
        self.datas = []
        
        for loc in loc_list[mode_to_loc[mode]]:
            img_loc_dir = os.path.join(self.img_path, loc)
            
            for img_loc_file in os.listdir(img_loc_dir):
                img_id = img_loc_file.split("_leftImg8bit.png")[0]
                
                data = {}
                data['img_path'] = os.path.join(img_loc_dir, img_loc_file)
                data['label_path'] = os.path.join(self.label_path, loc, img_id+'_gtFine_labelIds.png')
                data['img_id'] = img_id
                self.datas.append(data)
                
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        data = self.datas[index]
        img = Image.open(data['img_path']).convert('RGB')
        label = Image.open(data['label_path']).convert('RGB')
        if img is None:
            print(data['mg_path'])
        img, label = self.transform(img, label)
        
if __name__ == '__main__':
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    trans = [T.Resize((512, 1024))]
    if hflip_prob > 0:
        trans.append(T.RandomHorizontalFlip(hflip_prob))
    trans.extend([
        T.RandomCrop(crop_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    self.transforms = T.Compose(trans)

    
    dataset = CityScapes(transform=self.transforms)
    B = 2
    nw = 0
    trainDataloader = DataLoader(dataset, batch_size=B, shuffle=True, pin_memory=True, num_workers=nw)

    for img, label in trainDataloader:
        print(img.shape)
        print(label.shape)
        