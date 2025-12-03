from torch.utils.data import Dataset
import open3d as o3d
import numpy as np
import os
import torch

class ChairDataset(Dataset):
    def __init__(self, data_path, split="train", transforms=None):
        super(ChairDataset, self).__init__()
        self.split = split
        self.data_path = os.path.join(data_path, split)
        self.transforms = transforms
        self.pclouds_paths = os.listdir(os.path.join(self.data_path, "pts"))
        if os.path.exists(os.path.join(self.data_path, "label")):
            self.label_paths = os.listdir(os.path.join(self.data_path, "label"))
        else:
            self.label_paths = None

        self.class_dict = {
            0: "Armrest",
            1: "Backrest",
            2: "Chair legs",
            3: "Cushion"
        }

    def get_class_dict(self):
        return self.class_dict
        
    def __len__(self):
        return len(self.pclouds_paths)
    
    def __getitem__(self, idx):
        pclouds_path = self.pclouds_paths[idx]
        label_path = pclouds_path[:-3] + "txt"
        if self.label_paths is not None:
            pclouds_file = os.path.join(self.data_path, "pts", pclouds_path)
            label_file = os.path.join(self.data_path, "label", label_path)
            return self.load_pclouds(pclouds_file, label_file)
        else:
            return self.load_pclouds(os.path.join(self.data_path, pclouds_path))
            
    def load_pclouds(self, pclouds_file, label_path=None):
        pclouds = o3d.io.read_point_cloud(pclouds_file, format='xyz')
        pclouds = np.asarray(pclouds.points)

        if label_path is not None:
            with open(label_path, 'r') as file:
                labels = np.array([int(line.strip())-1 for line in file])
        else:
            labels = None

        if self.transforms:
            pclouds, labels = self.transforms(pclouds, labels)
        if labels is not None:
            return pclouds, labels
        else:
            return pclouds