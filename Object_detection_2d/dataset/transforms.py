import collections
import torch
import torchvision.transforms.functional as F
import random 
import numbers
import numpy as np
from PIL import Image
import torchvision.transforms.functional as FT

from Object_detection_2d.utils import find_boxes_overlap

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, img, target):
        if random.random() < self.p:
            img = F.hflip(img)
            w = img.shape[2]
            bboxes = target['bboxes']
            bboxes = bboxes.clone()
            
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]] - 1
            target['bboxes'] = bboxes

        return img, target
    
    def __repr__(self):
        return self.__class__.__name__ + 'p={}'.format(self.p)
    
class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = F.vflip(img)
            h = img.shape[1]
            bboxes = target['bboxes']
            bboxes = bboxes.clone()
            
            bboxes[:, [1, 3]] = h - bboxes[:, [3, 1]] - 1
            target['bboxes'] = bboxes

        return img, target

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
class RandomExpand(object):
    def __init__(self, filler, max_scale=4):
        self.max_scale = max_scale
        self.filler = torch.FloatTensor(filler)
        
    def __call__(self, img, target):
        
        if random.random() >= 0.5:
            return img, target
        
        h_ori = img.shape[1]
        w_ori = img.shape[2]
        scale = random.uniform(1, self.max_scale)
        h_new = int(scale * h_ori)
        w_new = int(scale * w_ori)
        
        new_image = torch.ones((3, h_new, w_new), dtype=torch.float32) * self.filler.unsqueeze(1).unsqueeze(1)
        left = random.randint(0, w_new - w_ori)
        right = left + w_ori
        top = random.randint(0, h_new - h_ori)
        bottom = top + h_ori
        new_image[:, top:bottom, left:right] = img
        
        new_boxes = target['bboxes'] + torch.FloatTensor([left, top, left, top]).unsqueeze(0)
        target['bboxes'] = new_boxes
        return new_image, target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, target):
        return F.normalize(img, self.mean, self.std), target

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class RandomCrop(object):
    def __init__(self):
        self.max_trials = 50
        self.min_overlap_list = [0., .1, .3, .5, .7, .9, None]
        self.min_scale = 0.3

    def __call__(self, img, target):
        h_ori, w_ori = img.shape[1:]
        
        while True:
            min_overlap = random.choice(self.min_overlap_list)
            
            if min_overlap is None:
                return img, target
            
            for _ in range(self.max_trials):
                h_scale = random.uniform(self.min_scale, 1)
                w_scale = random.uniform(self.min_scale, 1)
                
                h_new = int(h_scale * h_ori)
                w_new = int(w_scale * w_ori)
                
                aspect_ratio = h_new / w_new
                if not 0.5 < aspect_ratio < 2:
                    continue
                
                left = random.randint(0, w_ori - w_new)
                right = left + w_new
                top = random.randint(0, h_ori - h_new)
                bottom = top + h_new
                crop = torch.FloatTensor([left, top, right, bottom])
                
                overlap = find_boxes_overlap(crop.unsqueeze(0), target['bboxes']).squeeze(0)
                if overlap.max().item() < min_overlap:
                    continue
                
                img_new = img[:, top:bottom, left:right]
                bboxes = target['bboxes']
                labels = target['labels']
                difficulties = target['difficulties']
                centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2
                
                in_crop = (centers[:, 0] > left) * (centers[:, 0] < right) * (centers[:, 1] > top) * (centers[:, 1] < bottom)
                if not in_crop.any():
                    continue
                new_bboxes = bboxes[in_crop, :]
                new_labels = labels[in_crop]
                new_difficulties = difficulties[in_crop]
                new_bboxes[:, :2] = torch.max(new_bboxes[:, :2], crop[:2])
                new_bboxes[:, :2] -= crop[:2]
                new_bboxes[:, 2:] = torch.min(new_bboxes[:, 2:], crop[2:])
                new_bboxes[:, 2:] -= crop[:2]
                target['bboxes'] = new_bboxes
                target['labels'] = new_labels
                target['difficulties'] = new_difficulties
                return img_new, target

class Resize(object):
    def __init__(self, size, return_percent_coords=True):
        self.size = tuple(size)
        self.return_percent_coords = return_percent_coords

    def __call__(self, img, target):
        img_new = FT.resize(img, self.size)
        height, width = img.shape[1:]
        ori_size = torch.FloatTensor([width, height, width, height]).unsqueeze(0)
        bboxes = target['bboxes']
        new_bboxes = bboxes / ori_size
        
        if not self.return_percent_coords:
            new_sizes = torch.FloatTensor([self.size[1], self.size[0], self.size[1], self.size[0]]).unsqueeze(0)
            new_bboxes = new_bboxes * new_sizes
        target['bboxes'] = new_bboxes  
        return img_new, target

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, interpolation=bilinear)'.format(self.size)
    
    
class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                    clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = SingleCompose(transforms)

        return transform

    def __call__(self, img, label):
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img), label

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string
    

class Lambda(object):
    def __init__(self, lambd):
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ToTensor(object):
    def __init__(self, normalize=True, target_type='uint8'):
        self.normalize = normalize
        self.target_type = target_type
        
    def __call__(self, img, target):
        if self.normalize:
            return F.to_tensor(img), target
        else:
            return torch.from_numpy(np.array(img, dtype=np.float32).transpose(2, 0, 1) ), target

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
class SingleCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string