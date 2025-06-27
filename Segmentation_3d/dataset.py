import torch
from torch.utils.data import Dataset, DataLoader, random_split
import open3d as o3d
import numpy as np
import os
import random
import glob
import math
import json

def split_dataset_train_val(dataset: Dataset, split=0.9):
    train_size = int(split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

def normalize_pcloud(pcloud):
    centroid = np.mean(pcloud, axis=0)
    pcloud = pcloud - centroid
    pcloud = pcloud / np.mean(np.sqrt(np.sum(pcloud ** 2, axis=1)))
    return pcloud

def random_rotate_pcloud(pcloud):
    rot_angle = np.random.uniform() * 2 * np.pi
    sin, cos = np.sin(rot_angle), np.cos(rot_angle)
    rot_matrix = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
    pcloud = np.dot(pcloud, rot_matrix)
    return pcloud

def random_jitter_pcloud(pcloud, sigma=0.01):
    return pcloud + sigma * np.random.randn(*pcloud.shape)

def prepare_input(block_xyz, block_rgb, xcenter, ycenter, room_xyz_max):
    block_data = np.zeros([len(block_xyz), 9], dtype=np.float32)
    block_data[:, 0:3] = block_xyz[:, 0:3] - [xcenter, ycenter, 0]
    block_data[:, 3:6] = block_rgb / 255.0
    block_data[:, 6:9] = block_xyz / room_xyz_max
    return block_data

def random_dropout(indices, max_dropout=0.95):
    dropout = np.random.random() * max_dropout
    drop_idx = np.where(np.random.random(len(indices)) < dropout)[0]
    if len(drop_idx) > 0:
        indices[drop_idx] = indices[0]
    return indices

class ChairDataset(Dataset):
    def __init__(self, data_path, train=True, n_points=1500):
        super(ChairDataset, self).__init__()
        self.data_path = data_path
        self.train = train
        
        if self.train:
            self.data_path = os.path.join(self.data_path, "train")
            self.pcd_paths = os.listdir(os.path.join(self.data_path, "pts"))
            self.label_paths = os.listdir(os.path.join(self.data_path, "label"))
        else:
            self.data_path = os.path.join(self.data_path, "test")
            self.pcd_paths = os.listdir(self.data_path)
            self.label_paths = None
        self.n_points = n_points
        
    def __len__(self):
        return len(self.pcd_paths)
    
    def __getitem__(self, idx):
        pcd_path = self.pcd_paths[idx]
        label_path = pcd_path[:-3] + "txt"

        if self.train:
            pcloud, indices = self.load_pcloud(os.path.join(self.data_path, "pts", pcd_path))
            label = self.load_label(indices, os.path.join(self.data_path, "label", label_path))
            return pcloud.float(), label
        else:
            pcloud, _ = self.load_pcloud(os.path.join(self.data_path, pcd_path))
            return pcloud.float()
            
    def load_pcloud(self, pcdFile):
        pcd = o3d.io.read_point_cloud(pcdFile, format='xyz')
        points = np.asarray(pcd.points)
        indices = random.sample(list(range(points.shape[0])), self.n_points)
        points = points[indices, :]

        points = normalize_pcloud(points)
        points = torch.tensor(points, dtype=torch.float64).transpose(1, 0)
        return points, indices
    
    def load_label(self, indices, label_path):
        with open(label_path, 'r') as file:
            label = [int(line.strip())-1 for line in file]
        label = list(np.array(label)[indices])
        label = torch.tensor(label, dtype=torch.long)
        return label
    
class ModelNet40(Dataset):
    def __init__(self, data_path, n_points, split, random_rotate=False, random_jitter=False):
        super(ModelNet40, self).__init__()
        self.data_path = data_path
        self.split = split
        self.n_points = n_points
        self.random_rotate = random_rotate
        self.random_jitter = random_jitter
        
        self.class_dict = self.load_class_dict()
        self.id2name = list(self.class_dict.keys())
        self.model_indices = self.load_model_indices()
        
    def __len__(self):
        return len(self.model_indices)
    
    def __getitem__(self, index):
        return self.load_model(self.model_indices[index])
    
    def class_id2name(self, class_id):
        return self.id2name[class_id]
    
    def class_name2id(self, class_name):
        return self.class_dict[class_name]
    
    def load_class_dict(self):
        path = os.path.join(self.data_path, "modelnet40_shape_names.txt")
        class_names = [line.rstrip() for line in open(path)]
        return dict(zip(class_names, range(len(class_names))))
    
    def load_model_indices(self):
        path = os.path.join(self.data_path, 'modelnet40_%s.txt' % self.split)
        return [line.rstrip() for line in open(path)]
    
    def load_model(self, model_idx):
        class_name = '_'.join(model_idx.split('_')[0:-1])
        class_id = self.class_name2id(class_name)
        filepath = os.path.join(self.data_path, class_name, model_idx + '.npz')
        points = np.load(filepath)['xyz']

        if self.split == 'train':
            indices = np.random.choice(points.shape[0], self.n_points)
            points = points[indices, :]
            if self.random_rotate:
                points = random_rotate_pcloud(points)
            if self.random_jitter:
                points = random_jitter_pcloud(points)
        else:
            points = points[:self.n_points, :]

        points = normalize_pcloud(points)
        points = np.transpose(points)

        return points.astype(np.float32), class_id
    
class S3DIS_static(Dataset):
    def __init__(self, data_path, area_ids, n_points, max_dropout, offset_name=''):
        super(S3DIS_static, self).__init__()
        self.n_points = n_points
        self.max_dropout = max_dropout
        self.block_paths = []
        
        for area_id in area_ids:
            area_path = os.path.join(data_path, 'Area_{}'.format(area_id))
            for room_name in sorted(os.listdir(area_path)):
                room_path = os.path.join(area_path, room_name)
                for block_path in glob.glob(os.path.join(room_path, 'block_{}*.npz'.format(offset_name))):
                    n_points_in_block = np.load(block_path)['n_points_in_block']
                    self.block_paths.extend([block_path] * math.ceil(n_points_in_block / self.n_points))
                    
    def __len__(self):
        return len(self.block_paths)
    
    def __getitem__(self, index):
        data = np.load(self.block_paths[index])
        block_xyz, block_rgb, block_gt = data['block_xyz'], data['block_rgb'], data['block_gt']
        block_size, room_xyz_max = data['block_size'], data['room_xyz_max']

        xcenter, ycenter = np.amin(block_xyz, axis=0)[:2] + block_size / 2
        block_data = prepare_input(block_xyz, block_rgb, xcenter, ycenter, room_xyz_max)

        indices = np.random.choice(len(block_xyz), self.n_points, len(block_xyz) < self.n_points)
        indices = random_dropout(indices, self.max_dropout)

        block_data = block_data[indices].transpose()  # [n_channels, n_points]
        block_gt = block_gt[indices]

        return block_data.astype(np.float32), block_gt.astype(np.int64)

class S3DIS_dynamic(Dataset):
    def __init__(self, dataset_path, area_ids, n_points, max_dropout, block_size=1.0, sample_aug=1):
        super(S3DIS_dynamic, self).__init__()
        self.n_points = n_points
        self.max_dropout = max_dropout
        self.block_size = block_size
        self.rooms, self.indices = [], []
        
        for area_id in area_ids:
            area_path = os.path.join(dataset_path, 'Area_{}'.format(area_id))
            for room_path in glob.glob(os.path.join(area_path, '*_resampled.npz')):
                room_data = np.load(room_path)
                self.indices.extend([len(self.rooms)] * math.ceil(room_data['n_points'] / self.n_points) * sample_aug)
                self.rooms.append({
                    'xyz': room_data['xyz'],
                    'rgb': room_data['rgb'],
                    'gt': room_data['gt'],
                    'xyz_max': np.amax(room_data['xyz'], axis=0)
                })
                
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        room = self.rooms[self.indices[index]]
        room_xyz, room_rgb, room_gt, room_xyz_max = room['xyz'], room['rgb'], room['gt'], room['xyz_max']

        xcenter, ycenter = room_xyz[np.random.choice(room_xyz.shape[0])][:2]
        indices = self.get_block_indices(room_xyz, xcenter, ycenter)
        indices = random_dropout(indices, self.max_dropout)

        block_xyz, block_rgb, block_gt = room_xyz[indices], room_rgb[indices], room_gt[indices]
        block_data = prepare_input(block_xyz, block_rgb, xcenter, ycenter, room_xyz_max)
        block_data = np.transpose(block_data)  # [n_channels, n_points]

        return block_data.astype(np.float32), block_gt.astype(np.int64)
    
    def get_block_indices(self, room_xyz, xcenter, ycenter):
        xmin, xmax = xcenter - self.block_size / 2, xcenter + self.block_size / 2,
        ymin, ymax = ycenter - self.block_size / 2, ycenter + self.block_size / 2
        l, r = np.searchsorted(room_xyz[:, 0], [xmin, xmax])
        indices = np.where((room_xyz[l:r, 1] > ymin) & (room_xyz[l:r, 1] < ymax))[0] + l
        if len(indices) == 0:
            return indices, np.zeros([0, 9], dtype=np.float32), np.zeros([0, ], dtype=np.int64)
        if self.n_points != 'all':
            indices = np.random.choice(indices, self.n_points, indices.size < self.n_points)
        return indices

class S3DIS(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, split, test_area, n_points, max_dropout, block_type='dynamic', block_size=1.0):
        super().__init__()

        assert os.path.isdir(dataset_dir)
        assert split == 'train' or split == 'test'
        assert type(test_area) == int and 1 <= test_area <= 6
        assert 0 <= max_dropout <= 1

        area_ids = []
        for area_id in range(1, 7):
            if split == 'train' and area_id == test_area:
                continue
            if split == 'test' and area_id != test_area:
                continue
            area_ids.append(area_id)

        if block_type == 'static':
            offset_name = 'zero' if split == 'test' else ''
            self.dataset = S3DIS_static(dataset_dir, area_ids, n_points, max_dropout, offset_name)
        elif block_type == 'dynamic':
            sample_aug = 1 if split == 'test' else 2
            self.dataset = S3DIS_dynamic(dataset_dir, area_ids, n_points, max_dropout, block_size, sample_aug)
        else:
            raise NotImplementedError('Unknown block type: %s' % block_type)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
    
class ShapeNetPartDataset(Dataset):
    def __init__(self, root, n_points, split='train', class_choice=None, normal_channel=False):
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

        self.class2label = dict(zip(self.class2id, range(len(self.class2id))))

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
                
        self.seg_category = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
        self.cache = {}
        self.cache_size = 20000
        
    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]
        else:
            category, file_name = self.data_path[index]
            category_label = self.class2label[category]
            category_label = np.array([category_label]).astype(np.float32)
            data = np.loadtxt(file_name).astype(np.float32)
            if not self.normal_channel:
                pclouds = data[:, 0:3]
            else:
                pclouds = data[:, 0:6]
            seg_labels = data[:, -1].astype(np.int32)
            
            pclouds = normalize_pcloud(pclouds)
            sample_indexes = np.random.choice(len(seg_labels), self.n_points, replace=True)
            pclouds = pclouds[sample_indexes, :]
            seg_labels = seg_labels[sample_indexes]
            if len(self.cache) < self.cache_size:
                self.cache[index] = (pclouds, category_label, seg_labels)
            return (pclouds, category_label, seg_labels)
    
def get_dataset(args):
    dataset_type = args.dataset
    batch_size = args.batch_size
    n_points = args.n_points
    
    if dataset_type == "chair":
        path = os.path.join("Dataset", "Chair_dataset")
        train_dataset = ChairDataset(path, n_points=n_points)
        train_dataset, val_dataset = split_dataset_train_val(train_dataset)
        test_dataset = ChairDataset(path, train=False, n_points=n_points)
        class_dict = {
            0: "Armrest",
            1: "Backrest",
            2: "Chair legs",
            3: "Cushion"
        }
        
    elif dataset_type == "modelnet40":
        path = os.path.join("Dataset", "ModelNet40_npz")
        train_dataset = ModelNet40(path, n_points, "train")
        class_dict = train_dataset.id2name
        train_dataset, val_dataset = split_dataset_train_val(train_dataset)
        test_dataset = ModelNet40(path, n_points, "test")
        
    elif dataset_type == 's3dis':
        path = os.path.join("Dataset", "S3DIS_npz")
        test_area = args.test_area
        max_dropout = args.max_dropout
        block_type = args.block_type
        block_size = args.block_size
        
        train_dataset = S3DIS(path, "train", test_area, n_points, max_dropout, block_type, block_size)
        train_dataset, val_dataset = split_dataset_train_val(train_dataset)
        test_dataset = S3DIS(path, "test", test_area, n_points, max_dropout, block_type, block_size)
        class_dict = ['clutter', 'ceiling', 'floor', 'wall', 'beam', 'column', 'door',
               'window', 'table', 'chair', 'sofa', 'bookcase', 'board', 'stairs']
    else:
        raise ValueError(f'unknown dataset {dataset_type}')
        
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader, class_dict

if __name__ == '__main__':
    dataset = ShapeNetPartDataset("Dataset/ShapeNetPart", 1000)