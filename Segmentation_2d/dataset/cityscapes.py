from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

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
    19: "unlabeled"
}

label_color_dict = {
    0: [128, 64, 128],
    1: [244, 35, 232],
    2: [70, 70, 70],
    3: [102, 102, 156],
    4: [190, 153, 153],
    5: [153, 153, 153],
    6: [250, 170, 30],
    7: [220, 220, 0],
    8: [107, 142, 35],
    9: [152, 251, 152],
    10: [70, 130, 180],
    11: [220, 20, 60],
    12: [255, 0, 0],
    13: [0, 0, 142],
    14: [0, 0, 70],
    15: [0, 60, 100],
    16: [0, 80, 100],
    17: [0, 0, 230],
    18: [119, 11, 32],
    19: [0, 0, 0]
}


cityscapes_dataset_dirs = {
    "train": ["jena/", "zurich/", "weimar/", "ulm/", "tubingen/", "stuttgart/",
                "strasbourg/", "monchengladbach/", "krefeld/", "hanover/",
                "hamburg/", "erfurt/", "dusseldorf/", "darmstadt/", "cologne/",
                "bremen/", "bochum/", "aachen/"],
    "val": ["frankfurt/", "munster/", "lindau/"],
    "test": ["berlin", "bielefeld", "bonn", "leverkusen", "mainz", "munich"]
}


class CityScapesDataset(Dataset):
    
    train_id_to_color = [label_color_dict[c] for c in label_color_dict]
    train_id_to_color = np.array(train_id_to_color)
    def __init__(self, data_path, meta_path, split, transform=None):
        self.split = split
        self.img_dir = data_path + "leftImg8bit_trainvaltest/leftImg8bit/" + split
        self.label_dir = meta_path + "/label_imgs/"
        self.transform = transform
        
        self.img_h = 1024
        self.img_w = 2048
        self.new_img_h = 512
        self.new_img_w = 1024
        
        self.data = []
        for city in cityscapes_dataset_dirs[split]:
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
        img = Image.open(img_path).convert('RGB')
        
        label_path = data["label_img_path"]
        label = Image.open(label_path)
        
        if self.transform:
            img, label = self.transform(img, label)
        return img, label
    
    @classmethod
    def decode_target(cls, target):
        return cls.train_id_to_color[target]