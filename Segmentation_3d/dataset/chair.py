from torch.utils.data import Dataset
import open3d as o3d
import numpy as np
import os
import torch

from Segmentation_3d.dataset.transforms import normalize_pclouds, to_tensor, random_jitter_pclouds, \
    random_rotate_pclouds, random_shift_pclouds, random_scale_pclouds, get_fps_indexes

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
            pclouds_file = os.path.join(self.data_path, "pts", pclouds_path)
            label_file = os.path.join(self.data_path, "label", label_path)
            return self.load_pclouds(pclouds_file, label_file, transform=True)
        else:
            return self.load_pclouds(os.path.join(self.data_path, pclouds_path))
            
    def load_pclouds(self, pclouds_file, label_path=None, transform=False):
        pclouds = o3d.io.read_point_cloud(pclouds_file, format='xyz')
        pclouds = np.asarray(pclouds.points)
        pclouds = normalize_pclouds(pclouds)
        # if transform:
        #     pclouds, _ = random_rotate_pclouds(pclouds)
        #     pclouds = random_jitter_pclouds(pclouds)
        #     # pclouds = random_scale_pclouds(pclouds)
        #     # pclouds = random_shift_pclouds(pclouds)
            
        # Indexing
        farthest_indexes = get_fps_indexes(pclouds, self.n_points)
        pclouds = pclouds[farthest_indexes, :]
        pclouds = pclouds.transpose(1, 0)
        if label_path is None:
            pclouds, _ = to_tensor(pclouds)
            return pclouds

        with open(label_path, 'r') as file:
            labels = np.array([int(line.strip())-1 for line in file])
        labels = torch.from_numpy(labels).to(torch.long)
        labels = labels[farthest_indexes]
        return to_tensor(pclouds, labels)