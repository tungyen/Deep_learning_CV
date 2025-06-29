from torch.utils.data import Dataset
import open3d as o3d
import numpy as np
import os
import random
from Segmentation_3d.dataset.transforms import normalize_pclouds, to_tensor

class ChairDataset(Dataset):
    def __init__(self, data_path, train=True, n_points=1500):
        super(ChairDataset, self).__init__()
        self.data_path = data_path
        self.train = train
        if self.train:
            self.data_path = os.path.join(self.data_path, "train")
            self.pclouds_paths = os.listdir(os.path.join(self.data_path, "pts"))
            self.label_paths = os.listdir(os.path.join(self.data_path, "label"))
        else:
            self.data_path = os.path.join(self.data_path, "test")
            self.pclouds_paths = os.listdir(self.data_path)
            self.label_paths = None
        self.n_points = n_points
        
    def __len__(self):
        return len(self.pclouds_paths)
    
    def __getitem__(self, idx):
        pclouds_path = self.pclouds_paths[idx]
        label_path = pclouds_path[:-3] + "txt"

        if self.train:
            pclouds, indexes = self.load_pclouds(os.path.join(self.data_path, "pts", pclouds_path))
            seg_labels = self.load_labels(indexes, os.path.join(self.data_path, "label", label_path))
            return to_tensor(pclouds, seg_labels)
        else:
            pclouds, _ = self.load_pclouds(os.path.join(self.data_path, pclouds_path))
            return to_tensor(pclouds)
            
    def load_pclouds(self, pclouds_file):
        pclouds = o3d.io.read_point_cloud(pclouds_file, format='xyz')
        pclouds = np.asarray(pclouds.points)
        indexes = random.sample(list(range(pclouds.shape[0])), self.n_points)
        pclouds = pclouds[indexes, :]
        pclouds = normalize_pclouds(pclouds)
        pclouds = np.transpose(pclouds)
        return pclouds, indexes
    
    def load_labels(self, indexes, label_path):
        with open(label_path, 'r') as file:
            labels = [int(line.strip())-1 for line in file]
        labels = list(np.array(labels)[indexes])
        return labels