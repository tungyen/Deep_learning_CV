import numpy as np
import random
import torch

from Segmentation_3d.PointNet.model.utils import furthest_point_sampling

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pclouds, labels):
        for t in self.transforms:
            pclouds, labels = t(pclouds, labels)
        return pclouds, labels

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class NormalizePointClouds(object):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, pclouds, labels=None):
        centroid = np.mean(pclouds, axis=0)
        pclouds = pclouds - centroid
        pclouds = pclouds / np.mean(np.sqrt(np.sum(pclouds ** 2, axis=1)))
        return pclouds, labels

class TransposePointClouds(object):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, pclouds, labels=None):
        pclouds = np.transpose(pclouds)
        return pclouds, labels

class RandomScalePointClouds(object):
    def __init__(self, prob=0.05, scale_range=(0.75, 1.25)):
        self.scale_range = scale_range
        self.prob = prob

    def __call__(self, pclouds, labels=None):
        x = random.random()
        if x >= self.prob:
            return pclouds, labels
        scale = np.random.uniform(*self.scale_range)
        return pclouds * scale, labels

class RandomShiftPointClouds(object):
    def __init__(self, prob=0.05, shift_range=(-0.1, 0.1)):
        self.shift_range = shift_range
        self.prob = prob

    def __call__(self, pclouds, labels=None):
        x = random.random()
        if x >= self.prob:
            return pclouds, labels
        shift = np.random.uniform(self.shift_range[0], self.shift_range[1], size=(1, 3))
        return pclouds + shift, labels

class RandomRotatePointClouds(object):
    def __init__(self, prob=0.45):
        self.prob = prob

    def __call__(self, pclouds, labels=None):
        x = random.random()
        if x >= self.prob:
            return pclouds, labels
        rot_angle = np.random.uniform() * 2 * np.pi
        sin, cos = np.sin(rot_angle), np.cos(rot_angle)
        rot_matrix = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
        pclouds = np.dot(pclouds, rot_matrix)
        return pclouds, labels

class RandomJitterPointClouds(object):
    def __init__(self, prob=0.3, sigma=0.01, clip=0.02):
        self.sigma = sigma
        self.clip = clip
        self.prob = prob

    def __call__(self, pclouds, labels=None):
        x = random.random()
        if x >= self.prob:
            return pclouds, labels
        return pclouds + np.clip(self.sigma * np.random.randn(*pclouds.shape), -self.clip, self.clip)

class FPS(object):
    def __init__(self, num_points):
        self.num_points = num_points

    def __call__(self, pclouds, labels=None):
        pclouds_input = torch.from_numpy(pclouds).unsqueeze(0).to(torch.float32)
        fps_indexes = furthest_point_sampling(pclouds_input, self.num_points, cpp_impl=False).squeeze().cpu()
        pclouds = pclouds[fps_indexes, :]
        if labels is not None:
            labels = labels[fps_indexes]
        return pclouds, labels

class ToTensor(object):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, pclouds, labels=None):
        pclouds = torch.tensor(pclouds, dtype=torch.float32)
        if labels is not None:
            labels = torch.tensor(labels, dtype=torch.long)
        return pclouds, labels