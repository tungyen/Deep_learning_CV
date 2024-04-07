import torch
from torch.utils.data import Dataset
import open3d as o3d
import numpy as np
import os
import random

class chairDataset(Dataset):
    def __init__(self, dataPath, transform=None, pcdSize=1500):
        self.dataPath = dataPath
        self.pcdFiles = os.listdir(os.path.join(dataPath, "pts"))
        self.annotationFiles = os.listdir(os.path.join(dataPath, "label"))
        self.transform = transform
        self.pcdSize = pcdSize
        
    def __len__(self):
        return len(self.pcdFiles)
    
    def __getitem__(self, idx):
        pcdFile = self.pcdFiles[idx]
        annotationFile = self.annotationFiles[idx]
        
        pointcloud, indices = self.loadPCD(os.path.join(self.dataPath, "pts", pcdFile))
        annotation = self.loadAnnotation(indices, os.path.join(self.dataPath, "label", annotationFile))
        if self.transform is not None:
            pointcloud = self.transform(pointcloud)
        return pointcloud, annotation
    
    def loadPCD(self, pcdFile):
        # Inputs:
        #     pcdFile - The folder path of the point cloud data
        # Outputs:
        #     points - The point cloud data in numpy array with shape (n, 3)
        #     indices - The sampling index of the point cloud with shape (self.pcdSize, )
        pcd = o3d.io.read_point_cloud(pcdFile, format='xyz')
        points = np.asarray(pcd.points)
        
        # Sampling
        indices = random.sample(list(range(points.shape[0])), self.pcdSize)
        points = points[indices, :]
        
        mu = np.mean(points, axis=0)
        var = np.mean(np.square(points-mu))
        points = (points-mu) / np.sqrt(var)
        points = torch.tensor(points, dtype=torch.float32).transpose(1, 0)
        return points, indices
    
    def loadAnnotation(self, indices, annotationFile):
        # Inputs:
        #     indices - The sampling index of the point cloud with shape (self.pcdSize, )
        #     annotationFile - The folder path of the point cloud data
        # Outputs:
        #     annotation - The corresponding class of each point with tensor shape (n, 1)
        with open(annotationFile, 'r') as file:
            annotation = [int(line.strip())-1 for line in file]
        annotation = list(np.array(annotation)[indices])
        annotation = torch.tensor(annotation, dtype=torch.long)
        return annotation
    
    @staticmethod
    def collate_fn(batch):
        pcds, annotations = tuple(zip(*batch))
        images = torch.stack(pcds, dim=0)
        labels = torch.as_tensor(annotations)
        return images, labels
    
    
# if __name__ == '__main__':
#     dataPath = 'Dataset/train'
#     data = chairDataset(dataPath)
    
#     for d in data:
#         pcd, label = d
#         print(label.shape[0])