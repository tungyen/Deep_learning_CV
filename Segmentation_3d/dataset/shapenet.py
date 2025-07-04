from torch.utils.data import Dataset
import numpy as np
import os
import json

from Segmentation_3d.dataset.transforms import normalize_pclouds, to_tensor
    
class ShapeNetDataset(Dataset):
    def __init__(self, root, n_points, split='train', class_choice=None, normal_channel=False):
        super().__init__()
        self.n_points = n_points
        self.root = root
        self.category_file = os.path.join(root, 'synsetoffset2category.txt')
        self.class2id = {}
        self.normal_channel = normal_channel
        
        with open(self.category_file, 'r') as f:
            for line in f:
                line = line.strip().split()
                self.class2id[line[0]] = line[1]
        if class_choice is not None:
            self.class2id = {k: v for k, v in self.class2id.items() if k in class_choice}

        self.meta = {}
        assert split in ["train", "val", "test"], "Unknown split is used."
        with open(os.path.join(self.root, 'train_test_split', f'shuffled_{split}_file_list.json'), 'r') as f:
            data_ids = set([str(d.split('/')[2]) for d in json.load(f)])
            
        for category in self.class2id:
            self.meta[category] = []
            pclouds_dir = os.path.join(self.root, self.class2id[category])
            pclouds_files = sorted(os.listdir(pclouds_dir))
            
            pclouds_files = [file for file in pclouds_files if ((file[0:-4] in data_ids))]
            for pclouds_file in pclouds_files:
                file_name = (os.path.splitext(os.path.basename(pclouds_file))[0])
                self.meta[category].append(os.path.join(pclouds_dir, file_name + '.txt'))
                
        self.data_path = []
        for category in self.class2id:
            for file_name in self.meta[category]:
                self.data_path.append((category, file_name))
        self.cache = {}
        self.cache_size = 20000
        
        self.instance2parts = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
        self.parts2instance = {}
        for cls in self.instance2parts.keys():
            for label in self.instance2parts[cls]:
                self.parts2instance[label] = cls

        self.class2label = dict(zip(self.class2id, range(len(self.class2id))))
        
    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]
        else:
            cls_names, file_name = self.data_path[index]
            cls_labels = self.class2label[cls_names]
            data = np.loadtxt(file_name).astype(np.float32)
            if not self.normal_channel:
                pclouds = data[:, 0:3]
            else:
                pclouds = data[:, 0:6]
            seg_labels = data[:, -1]
            pclouds = normalize_pclouds(pclouds)
            pclouds = np.transpose(pclouds)
            sample_indexes = np.random.choice(pclouds.shape[1], self.n_points, replace=True)
            pclouds = pclouds[:, sample_indexes]
            seg_labels = seg_labels[sample_indexes]
            pclouds, cls_labels = to_tensor(pclouds, cls_labels)
            pclouds, seg_labels = to_tensor(pclouds, seg_labels)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (pclouds, cls_labels, seg_labels)
            return (pclouds, cls_labels, seg_labels)