import numpy as np
import torch

def normalize_pclouds(pclouds):
    centroid = np.mean(pclouds, axis=0)
    pclouds = pclouds - centroid
    pclouds = pclouds / np.mean(np.sqrt(np.sum(pclouds ** 2, axis=1)))
    return pclouds

def random_rotate_pclouds(pclouds):
    rot_angle = np.random.uniform() * 2 * np.pi
    sin, cos = np.sin(rot_angle), np.cos(rot_angle)
    rot_matrix = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
    pclouds = np.dot(pclouds, rot_matrix)
    return pclouds

def random_jitter_pclouds(pclouds, sigma=0.01):
    return pclouds + sigma * np.random.randn(*pclouds.shape)

def random_dropout(indices, max_dropout=0.95):
    dropout = np.random.random() * max_dropout
    drop_idx = np.where(np.random.random(len(indices)) < dropout)[0]
    if len(drop_idx) > 0:
        indices[drop_idx] = indices[0]
    return indices

def to_tensor(pclouds, labels=None):
    pclouds = torch.tensor(pclouds, dtype=torch.float32)
    if labels is not None:
        labels = torch.tensor(labels, dtype=torch.long)
    return pclouds, labels