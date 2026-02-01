import collections
import torch
import torchvision.transforms.functional as F
import random 
import numbers
import numpy as np
from PIL import Image
import cv2

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, input_dict):
        for t in self.transforms:
            input_dict = t(input_dict)
        return input_dict
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, input_dict: dict):
        if random.random() < self.prob:
            input_dict['img'] = input_dict['img'][:, ::-1, :]
            input_dict['label'] = input_dict['label'][:, ::-1]
            return input_dict
        return input_dict
    
    def __repr__(self):
        return self.__class__.__name__ + 'p={}'.format(self.prob)
    
class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, input_dict: dict):
        if random.random() < self.prob:
            input_dict['img'] = F.vflip(input_dict['img'])
            input_dict['label'] = F.vflip(input_dict['label'])
            return input_dict
        return input_dict

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.prob)
    
    
class CenterCrop(object):
    def __init__(self, crop_size):
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = tuple(crop_size)

    def __call__(self, input_dict: dict):
        input_dict['img'] = F.center_crop(input_dict['img'], self.crop_size)
        input_dict['label'] = F.center_crop(input_dict['label'], self.crop_size)
        return input_dict

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.crop_size)
    
    
class RandomScale(object):
    def __init__(self, scale_range, interpolation=Image.BILINEAR):
        self.scale_range = scale_range
        self.interpolation = interpolation

    def __call__(self, input_dict: dict):
        img = input_dict['img']
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        target_size = ( int(img.shape[0]*scale), int(img.shape[1]*scale))
        input_dict['img'] = F.resize(img, target_size, self.interpolation)
        input_dict['label'] = F.resize(label, target_size, Image.NEAREST)
        return input_dict

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, interpolation=bilinear)'.format(self.size)
    

class Scale(object):
    def __init__(self, scale, interpolation=Image.BILINEAR):
        self.scale = scale
        self.interpolation = interpolation

    def __call__(self, input_dict: dict):
        img = input_dict['img']
        label = input_dict['label']
        assert img.size == label.size
        target_size = ( int(img.shape[0]*self.scale), int(img.shape[1]*self.scale)) # (H, W)
        input_dict['img'] = F.resize(img, target_size, self.interpolation)
        input_dict['label'] = F.resize(label, target_size, Image.NEAREST)
        return input_dict

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, interpolation=bilinear)'.format(self.size)

    
class Pad(object):
    def __init__(self, diviser=32):
        self.diviser = diviser
    
    def __call__(self, input_dict: dict):
        img = input_dict['img']
        label = input_dict['label']
        h, w = img.size
        ph = (h//self.diviser+1)*self.diviser - h if h%self.diviser!=0 else 0
        pw = (w//self.diviser+1)*self.diviser - w if w%self.diviser!=0 else 0
        input_dict['img'] = F.pad(img, (pw//2, pw-pw//2, ph//2, ph-ph//2))
        input_dict['label'] = F.pad(label, (pw//2, pw-pw//2, ph//2, ph-ph//2))
        return input_dict
    
    
class ToTensor(object):
    def __init__(self, normalize=True):
        self.normalize = normalize
        
    def __call__(self, input_dict: dict):
        img = input_dict['img']
        label = input_dict['label']
        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
        if self.normalize:
            img = img.float() / 255.0
        else:
            img = img.float()
        label = torch.from_numpy(label.astype(np.int64)).long()
        input_dict['img'] = img
        input_dict['label'] = label
        for k in input_dict:
            if isinstance(input_dict[k], list):
                input_dict[k] = torch.tensor(input_dict[k])
        return input_dict

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
    
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, input_dict: dict):
        img = input_dict['img'].float()
        input_dict['img'] = F.normalize(img, self.mean, self.std)
        return input_dict

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
    
class RandomCrop(object):
    def __init__(self, crop_size, ignore_index, padding=0, pad_if_needed=False):
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.ignore_index = ignore_index

    @staticmethod
    def get_params(img, output_size):
        h, w = img.shape[0], img.shape[1]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, input_dict: dict):
        img = input_dict['img']
        label = input_dict['label']

        h, w = img.shape[:2]
        crop_h, crop_w = self.crop_size

        # pad the width if needed
        if self.pad_if_needed and w < crop_w:
            pad = int((1 + crop_w - w) / 2)
            img = np.pad(img, ((0, 0), (pad, pad), (0, 0)), mode='constant', constant_value=0)
            label = np.pad(label, ((0, 0), (pad, pad), (0, 0)), mode='constant', constant_value=self.ignore_index)

        # pad the height if needed
        if self.pad_if_needed and h < crop_h:
            pad = int((1 + crop_h - h) / 2)
            img = np.pad(img, ((pad, pad), (0, 0), (0, 0)), mode='constant', constant_value=0)
            label = np.pad(label, ((pad, pad), (0, 0), (0, 0)), mode='constant', constant_value=self.ignore_index)

        i, j, h, w = self.get_params(img, self.crop_size)
        input_dict['img'] = img[i:i+h, j:j+w, :]
        input_dict['label'] = label[i:i+h, j:j+w]
        return input_dict

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.crop_size, self.padding)
    
    
class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = tuple(size)
        self.interpolation = interpolation

    def __call__(self, input_dict: dict):
        img = input_dict['img']
        label = input_dict['label']
        h, w = img.shape[0], img.shape[1]
        input_dict['img'] = F.resize(img, self.size, self.interpolation)
        input_dict['label'] = F.resize(label, self.size, Image.NEAREST)
        input_dict['original_size'] = [h, w]
        return input_dict

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, interpolation=bilinear)'.format(self.size)

class ResizeShorterSide(object):
    def __init__(self, shorter_size):
        self.shorter_size = shorter_size

    def __call__(self, input_dict: dict):
        img = input_dict['img']
        label = input_dict['label']
        h, w = img.shape[0], img.shape[1]
        if w < h:
            target_size = (self.shorter_size, int(h/w * self.shorter_size))
        else:
            target_size = (int(w/h * self.shorter_size), self.shorter_size)
        input_dict['img'] = cv2.resize(img, target_size)

        label = input_dict['label']
        label = Image.fromarray(label)
        target_w, target_h = target_size
        label = np.array(F.resize(label, (target_h, target_w), Image.NEAREST))
        input_dict['label'] = label
        return input_dict

class ResizeLetterBoxes(object):
    def __init__(self, size, ignore_index):
        self.size = size
        self.ignore_index = ignore_index

    def __call__(self, input_dict: dict):
        if isinstance(self.size, tuple):
            ih, iw = self.size[1], self.size[0]
        else:
            ih, iw = self.size, self.size
        img = input_dict['img']
        label = input_dict['label']
        h, w = img.shape[0], img.shape[1]
        scale = min(iw/w, ih/h)
        nw, nh = int(scale*w), int(scale*h)
        img_resized = cv2.resize(img, (nw, nh))
        label_resized = cv2.resize(label, (nw, nh), interpolation=cv2.INTER_NEAREST)
        dw, dh = (iw-nw) // 2, (ih-nh) // 2
        img_padded = np.full(shape=[ih, iw, 3], fill_value=0)
        img_padded[dh:nh+dh, dw:dw+nw, :] = img_resized
        label_padded = np.full(shape=[ih, iw], fill_value=self.ignore_index)
        label_padded[dh:nh+dh, dw:dw+nw] = label_resized
        input_dict['img'] = img_padded
        input_dict['label'] = label_padded
        input_dict['scale'] = scale
        input_dict['padding'] = [dw, dh]
        input_dict['rescale_size'] = [nw, nh]
        input_dict['original_size'] = [h, w]
        return input_dict
    
    
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

    def __call__(self, input_dict: dict):
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        input_dict['img'] = transform(input_dict['img'])
        return input_dict

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string