import collections
import torch
import torchvision.transforms.functional as F
import random 
import numbers
import numpy as np
from PIL import Image

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label
    
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
        
    def __call__(self, img, label):
        if random.random() < self.prob:
            return F.hflip(img), label
        return img, label
    
    def __repr__(self):
        return self.__class__.__name__ + 'p={}'.format(self.prob)
    
class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, label):
        if random.random() < self.prob:
            return F.vflip(img), label
        return img, label

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.prob)


class RandomResizeCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3./4., 4./3.), interpolation=Image.BILINEAR):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    @staticmethod
    def get_params(img, scale, ratio):
        width, height = img.size
        area = height * width

        for _ in range(10):
            target_area = random.uniform(scale[0], scale[1]) * area
            log_ratio = (np.log(ratio[0]), np.log(ratio[1]))
            aspect_ratio = np.exp(random.uniform(log_ratio[0], log_ratio[1]))

            w = int(round(np.sqrt(target_area * aspect_ratio)))
            h = int(round(np.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to center crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img, label):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = F.crop(img, i, j, h, w)
        img = F.resize(img, self.size, self.interpolation)
        return img, label

class CenterCrop(object):
    def __init__(self, crop_size):
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = tuple(crop_size)

    def __call__(self, img, label):
        return F.center_crop(img, self.crop_size), label

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.crop_size)
    
    
class RandomScale(object):
    def __init__(self, scale_range, interpolation=Image.BILINEAR):
        self.scale_range = scale_range
        self.interpolation = interpolation

    def __call__(self, img, label):
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        target_size = ( int(img.size[1]*scale), int(img.size[0]*scale) )
        return F.resize(img, target_size, self.interpolation), label

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, interpolation=bilinear)'.format(self.size)
    

class Scale(object):
    def __init__(self, scale, interpolation=Image.BILINEAR):
        self.scale = scale
        self.interpolation = interpolation

    def __call__(self, img, label):
        assert img.size == label.size
        target_size = ( int(img.size[1]*self.scale), int(img.size[0]*self.scale) ) # (H, W)
        return F.resize(img, target_size, self.interpolation), label

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, interpolation=bilinear)'.format(self.size)
    
    
class RandomRotation(object):
    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, img, label):
        angle = self.get_params(self.degrees)
        return F.rotate(img, angle, self.resample, self.expand, self.center), label

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string
    
class Pad(object):
    def __init__(self, diviser=32):
        self.diviser = diviser
    
    def __call__(self, img, label):
        h, w = img.size
        ph = (h//self.diviser+1)*self.diviser - h if h%self.diviser!=0 else 0
        pw = (w//self.diviser+1)*self.diviser - w if w%self.diviser!=0 else 0
        img = F.pad(img, ( pw//2, pw-pw//2, ph//2, ph-ph//2) )
        return img, label
    
    
class ToTensor(object):
    def __init__(self, normalize=True, target_type='uint8'):
        self.normalize = normalize
        self.target_type = target_type
        
    def __call__(self, img, label):
        if self.normalize:
            return F.to_tensor(img), torch.from_numpy( np.array( label, dtype=self.target_type) )
        else:
            return torch.from_numpy( np.array(img, dtype=np.float32).transpose(2, 0, 1) ), torch.from_numpy( np.array( label, dtype=self.target_type) )

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
    
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, label):
        return F.normalize(tensor, self.mean, self.std), label

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
    
class RandomCrop(object):
    def __init__(self, crop_size, padding=0, pad_if_needed=False):
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, label):
        if self.padding > 0:
            img = F.pad(img, self.padding)
            label = F.pad(label, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.crop_size[1]:
            img = F.pad(img, padding=int((1 + self.crop_size[1] - img.size[0]) / 2))

        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.crop_size[0]:
            img = F.pad(img, padding=int((1 + self.crop_size[0] - img.size[1]) / 2))

        i, j, h, w = self.get_params(img, self.crop_size)

        return F.crop(img, i, j, h, w), label

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.crop_size, self.padding)
    
    
class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = tuple(size)
        self.interpolation = interpolation

    def __call__(self, img, label):
        return F.resize(img, self.size, self.interpolation), label

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