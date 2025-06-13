import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import cv2
import os
import random
from torchvision import transforms
from PIL import Image

cityscapes_class_dict = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
    19: "unlabeled"  # often used for ignore index in semantic segmentation
}


dataset_dirs = {
    "train": ["jena/", "zurich/", "weimar/", "ulm/", "tubingen/", "stuttgart/",
                "strasbourg/", "monchengladbach/", "krefeld/", "hanover/",
                "hamburg/", "erfurt/", "dusseldorf/", "darmstadt/", "cologne/",
                "bremen/", "bochum/", "aachen/"],
    "val": ["frankfurt/", "munster/", "lindau/"],
    "test": ["berlin", "bielefeld", "bonn", "leverkusen", "mainz", "munich"]
}

train_dirs = ["jena/", "zurich/", "weimar/", "ulm/", "tubingen/", "stuttgart/",
                "strasbourg/", "monchengladbach/", "krefeld/", "hanover/",
                "hamburg/", "erfurt/", "dusseldorf/", "darmstadt/", "cologne/",
                "bremen/", "bochum/", "aachen/"]
val_dirs = ["frankfurt/", "munster/", "lindau/"]
test_dirs = ["berlin", "bielefeld", "bonn", "leverkusen", "mainz", "munich"]

class CityScapesDataset(Dataset):
    def __init__(self, data_path, meta_path, split):
        self.split = split
        self.img_dir = data_path + "leftImg8bit_trainvaltest/leftImg8bit/" + split
        self.label_dir = meta_path + "/label_imgs/"
        
        self.img_h = 1024
        self.img_w = 2048
        self.new_img_h = 512
        self.new_img_w = 1024
        
        self.data = []
        for city in dataset_dirs[split]:
            city_path = os.path.join(self.img_dir, city)
            city_img_names = os.listdir(city_path)
            
            for city_img_name in city_img_names:
                img_path = os.path.join(city_path, city_img_name)
                img_id = city_img_name.split("_leftImg8bit.png")[0]
                label_img_path = os.path.join(self.label_dir, img_id+".png")

                sample = {}
                sample["img_path"] = img_path
                sample["label_img_path"] = label_img_path
                sample["img_id"] = img_id
                self.data.append(sample)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        data = self.data[index]

        img_path = data["img_path"]
        img = cv2.imread(img_path, -1)
        # img = cv2.resize(img, (self.new_img_w, self.new_img_h), interpolation=cv2.INTER_NEAREST)
        
        label_img_path = data["label_img_path"]
        label_img = cv2.imread(label_img_path, -1)
        # label_img = cv2.resize(label_img, (self.new_img_w, self.new_img_h), interpolation=cv2.INTER_NEAREST)

        flip = np.random.randint(low=0, high=2)
        if flip == 1:
            img = cv2.flip(img, 1)
            label_img = cv2.flip(label_img, 1)
        new_img_h = self.img_h
        new_img_w = self.img_w
        scale = np.random.uniform(low=0.7, high=2.0)
        new_img_h = int(scale*new_img_h)
        new_img_w = int(scale*new_img_w)
        img = cv2.resize(img, (new_img_w, new_img_h), interpolation=cv2.INTER_NEAREST)
        label_img = cv2.resize(label_img, (new_img_w, new_img_h), interpolation=cv2.INTER_NEAREST)

        start_x = np.random.randint(low=0, high=(new_img_w - 513))
        end_x = start_x + 513
        start_y = np.random.randint(low=0, high=(new_img_h - 513))
        end_y = start_y + 513

        img = img[start_y:end_y, start_x:end_x]
        label_img = label_img[start_y:end_y, start_x:end_x]

        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32)

        img = torch.from_numpy(img)
        label_img = torch.from_numpy(label_img)
        return (img, label_img)


def get_dataset(args):
    dataset_type = args.dataset
    batch_size = args.batch_size
    
    if dataset_type == "cityscapes":
        data_path = "../../Dataset/cityscapes/"
        meta_path = "../../Dataset/cityscapes/meta"
        
        train_dataset = CityScapesDataset(data_path, meta_path, "train")
        val_dataset = CityScapesDataset(data_path, meta_path, "val")
        test_dataset = CityScapesDataset(data_path, meta_path, "test")
        class_dict = cityscapes_class_dict
    else:
        raise ValueError(f'Unknown dataset {dataset_type}.')
        
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_dataloader, val_dataloader, test_dataloader, class_dict