import numpy as np
import torch

from Segmentation_3d.PointNet.model.utils import furthest_point_sampling

def normalize_pclouds(pclouds):
    centroid = np.mean(pclouds, axis=0)
    pclouds = pclouds - centroid
    pclouds = pclouds / np.mean(np.sqrt(np.sum(pclouds ** 2, axis=1)))
    return pclouds

def random_scale_pclouds(pclouds, scale_range=(0.75, 1.25)):
    scale = np.random.uniform(*scale_range)
    return pclouds * scale

def random_shift_pclouds(pclouds, shift_range=(-0.1, 0.1)):
    shift = np.random.uniform(shift_range[0], shift_range[1], size=(1, 3))
    return pclouds + shift

def random_rotate_pclouds(pclouds):
    rot_angle = np.random.uniform() * 2 * np.pi
    sin, cos = np.sin(rot_angle), np.cos(rot_angle)
    rot_matrix = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
    pclouds = np.dot(pclouds, rot_matrix)
    return pclouds, rot_matrix

def random_jitter_pclouds(pclouds, sigma=0.01, clip=0.05):
    return pclouds + np.clip(sigma * np.random.randn(*pclouds.shape), -clip, clip)

def random_dropout(indices, max_dropout=0.95):
    dropout = np.random.random() * max_dropout
    drop_idx = np.where(np.random.random(len(indices)) < dropout)[0]
    if len(drop_idx) > 0:
        indices[drop_idx] = indices[0]
    return indices

def get_fps_indexes(pclouds, n_points):
    pclouds_cuda_input = torch.from_numpy(pclouds).unsqueeze(0).cuda().to(torch.float32)
    fps_indexes = furthest_point_sampling(pclouds_cuda_input, n_points).squeeze().cpu()
    return fps_indexes

def to_tensor(pclouds, labels=None):
    pclouds = torch.tensor(pclouds, dtype=torch.float32)
    if labels is not None:
        labels = torch.tensor(labels, dtype=torch.long)
    return pclouds, labels