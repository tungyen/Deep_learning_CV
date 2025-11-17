from itertools import product
import torch
from math import sqrt

class PriorBox:
    def __init__(self, img_size, feature_map_sizes=[], min_sizes=[],
                 max_sizes=[], strides=[], aspect_ratios=[], clip=True):
        self.img_size = img_size
        self.feature_map_sizes = feature_map_sizes
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.strides = strides
        self.aspect_ratios = aspect_ratios
        self.clip = clip

    def __call__(self):
        priors = []
        for k, f in enumerate(self.feature_map_sizes):
            scale = self.img_size / self.strides[k]
            for i, j in product(range(f), repeat=2):
                cx = (j + 0.5) / scale
                cy = (i + 0.5) / scale

                size = self.min_sizes[k]
                h = w = size / self.img_size
                priors.append([cx, cy, w, h])

                size = sqrt(self.min_sizes[k] * self.max_sizes[k])
                h = w = size / self.img_size
                priors.append([cx, cy, w, h])

                size = self.min_sizes[k]
                h = w = size / self.img_size
                for ratio in self.aspect_ratios[k]:
                    ratio = sqrt(ratio)
                    priors.append([cx, cy, w * ratio, h / ratio])
                    priors.append([cx, cy, w / ratio, h * ratio])
        
        priors = torch.tensor(priors)
        if self.clip:
            priors.clamp_(max=1, min=0)
        return priors