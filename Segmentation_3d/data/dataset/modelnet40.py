from torch.utils.data import Dataset
import numpy as np
import os
import csv

class ModelNet40Dataset(Dataset):
    def __init__(self, data_path, split, transforms=None):
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.csv_path = os.path.join(data_path, "metadata_modelnet40.csv")
        self.class_dict = self.load_class_dict()
        self.id2name = list(self.class_dict.keys())
        self.model_indexes = self.load_model_indexes()
        self.transforms = transforms
        
    def __len__(self):
        return len(self.model_indexes)
    
    def __getitem__(self, index):
        return self.load_model(self.model_indexes[index])
    
    def class_id2name(self, class_id):
        return self.id2name[class_id]
    
    def class_name2id(self, class_name):
        return self.class_dict[class_name]

    def get_class_dict(self):
        return self.id2name
    
    def load_class_dict(self):
        classes = set()
        with open(self.csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                classes.add(row["class"])
        classes = sorted(list(classes))
        class_dict = dict(zip(classes, range(len(classes))))
        class_dict["flower_pot"] = class_dict["flower"]
        class_dict.pop("flower")
        class_dict["glass_box"] = class_dict["glass"]
        class_dict.pop("glass")
        class_dict["night_stand"] = class_dict["night"]
        class_dict.pop("night")
        class_dict["range_hood"] = class_dict["range"]
        class_dict.pop("range")
        class_dict["tv_stand"] = class_dict["tv"]
        class_dict.pop("tv")
        return class_dict
    
    def load_model_indexes(self):
        path = os.path.join(self.data_path, self.split)
        data_list = []
        for class_name in os.listdir(path):
            class_path = os.path.join(path, class_name)
            for pclouds_name in os.listdir(class_path):
                pclouds_path = os.path.join(class_path, pclouds_name)
                data_list.append(pclouds_path)
        return data_list
    
    def load_model(self, model_idx):
        class_name = '_'.join(os.path.basename(model_idx).split('_')[0:-1])
        cls_labels = self.class_name2id(class_name)
        pclouds = np.load(model_idx)['xyz'].astype(np.float32)

        if self.transforms is not None:
            pclouds, cls_labels = self.transforms(pclouds, cls_labels)
        return pclouds, cls_labels