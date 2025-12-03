from torch.utils.data import Dataset
import numpy as np
import os

class ModelNet40Dataset(Dataset):
    def __init__(self, data_path, n_points, split):
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.n_points = n_points
        self.class_dict = self.load_class_dict()
        self.id2name = list(self.class_dict.keys())
        self.model_indexes = self.load_model_indexes()
        
    def __len__(self):
        return len(self.model_indexes)
    
    def __getitem__(self, index):
        return self.load_model(self.model_indexes[index])
    
    def class_id2name(self, class_id):
        return self.id2name[class_id]
    
    def class_name2id(self, class_name):
        return self.class_dict[class_name]
    
    def load_class_dict(self):
        path = os.path.join(self.data_path, "modelnet40_shape_names.txt")
        class_names = [line.rstrip() for line in open(path)]
        return dict(zip(class_names, range(len(class_names))))
    
    def load_model_indexes(self):
        path = os.path.join(self.data_path, 'modelnet40_%s.txt' % self.split)
        return [line.rstrip() for line in open(path)]
    
    def load_model(self, model_idx):
        class_name = '_'.join(model_idx.split('_')[0:-1])
        cls_labels = self.class_name2id(class_name)
        filepath = os.path.join(self.data_path, class_name, model_idx + '.npz')
        pclouds = np.load(filepath)['xyz'].astype(np.float32)
        pclouds = normalize_pclouds(pclouds)
        
        farthest_indexes = get_fps_indexes(pclouds, self.n_points)
        pclouds = pclouds[farthest_indexes, :]

        pclouds = np.transpose(pclouds)
        return to_tensor(pclouds, cls_labels)