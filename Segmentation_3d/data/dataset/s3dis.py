from torch.utils.data import Dataset
import numpy as np
import os
import glob
import math

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

class S3disStatic(Dataset):
    def __init__(self, data_path, area_ids, n_points, max_dropout, offset_name=''):
        super().__init__()
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

        farthest_indexes = get_fps_indexes(block_data[:, :3], self.n_points)
        farthest_indexes = random_dropout(farthest_indexes.numpy(), self.max_dropout)

        pclouds = block_data[farthest_indexes].transpose()
        seg_labels = block_gt[farthest_indexes]
        return to_tensor(pclouds, seg_labels)

class S3disDynamic(Dataset):
    def __init__(self, dataset_path, area_ids, n_points, max_dropout, block_size=1.0, sample_aug=1):
        super().__init__()
        self.n_points = n_points
        self.max_dropout = max_dropout
        self.block_size = block_size
        self.rooms, self.indexes = [], []
        
        for area_id in area_ids:
            area_path = os.path.join(dataset_path, 'Area_{}'.format(area_id))
            for room_path in glob.glob(os.path.join(area_path, '*_resampled.npz')):
                room_data = np.load(room_path)
                self.indexes.extend([len(self.rooms)] * math.ceil(room_data['n_points'] / self.n_points) * sample_aug)
                self.rooms.append({
                    'xyz': room_data['xyz'],
                    'rgb': room_data['rgb'],
                    'gt': room_data['gt'],
                    'xyz_max': np.amax(room_data['xyz'], axis=0)
                })
                
    def __len__(self):
        return len(self.indexes)
    
    def __getitem__(self, index):
        room = self.rooms[self.indexes[index]]
        room_xyz, room_rgb, room_gt, room_xyz_max = room['xyz'], room['rgb'], room['gt'], room['xyz_max']

        xcenter, ycenter = room_xyz[np.random.choice(room_xyz.shape[0])][:2]
        indexes = self.get_block_indexes(room_xyz, xcenter, ycenter)
        indexes = random_dropout(indexes, self.max_dropout)

        block_xyz, block_rgb, seg_labels = room_xyz[indexes], room_rgb[indexes], room_gt[indexes]
        block_data = prepare_input(block_xyz, block_rgb, xcenter, ycenter, room_xyz_max)
        pclouds = np.transpose(block_data)
        return to_tensor(pclouds, seg_labels)
    
    def get_block_indexes(self, room_xyz, xcenter, ycenter):
        xmin, xmax = xcenter - self.block_size / 2, xcenter + self.block_size / 2,
        ymin, ymax = ycenter - self.block_size / 2, ycenter + self.block_size / 2
        l, r = np.searchsorted(room_xyz[:, 0], [xmin, xmax])
        indexes = np.where((room_xyz[l:r, 1] > ymin) & (room_xyz[l:r, 1] < ymax))[0] + l
        if len(indexes) == 0:
            return indexes, np.zeros([0, 9], dtype=np.float32), np.zeros([0, ], dtype=np.int64)
        if self.n_points != 'all':
            indexes = np.random.choice(indexes, self.n_points, indexes.size < self.n_points)
        return indexes

class S3disDataset(Dataset):
    def __init__(self, dataset_dir, split, n_points, test_area=5, max_dropout=0.95, block_type='static', block_size=1.0):
        super().__init__()
        self.class_list = ['clutter', 'ceiling', 'floor', 'wall', 'beam', 'column', 'door',
                           'window', 'table', 'chair', 'sofa', 'bookcase', 'board', 'stairs']
        self.class_dict = {i: name for i, name in enumerate(self.class_list)}

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
            self.dataset = S3disStatic(dataset_dir, area_ids, n_points, max_dropout, offset_name)
        elif block_type == 'dynamic':
            sample_aug = 1 if split == 'test' else 2
            self.dataset = S3disDynamic(dataset_dir, area_ids, n_points, max_dropout, block_size, sample_aug)
        else:
            raise NotImplementedError('Unknown block type: %s' % block_type)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def get_class_dict(self):
        return self.class_dict