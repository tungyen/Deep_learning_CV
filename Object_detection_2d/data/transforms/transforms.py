import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random

from core.utils.box_utils import *

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, input_dict: dict):
        for t in self.transforms:
            input_dict = t(input_dict)
            boxes = input_dict.get('boxes', None)
            if boxes is not None:
                input_dict['boxes'], input_dict['labels'] = remove_empty_boxes(input_dict['boxes'], input_dict['labels'])
        return input_dict

class ConvertFromInts(object):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, input_dict: dict):
        img = input_dict['img']
        input_dict['img'] = img.astype(np.float32)
        return input_dict

class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, input_dict: dict):
        img = input_dict['img'].astype(np.float32)
        img -= self.mean
        input_dict['img'] = img
        return input_dict

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, input_dict: dict):
        img = input_dict['img'].astype(np.float32)
        if np.max(img) > 1.0:
            img /= 255.0
        img -= self.mean
        img /= self.std
        input_dict['img'] = img.astype(np.float32)
        return input_dict

class ToAbsoluteCoords(object):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, input_dict: dict):
        img = input_dict['img']
        boxes = input_dict.get('boxes', None)

        if boxes is None:
            return input_dict
        height, width, channels = img.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height
        input_dict['boxes'] = boxes
        return input_dict

class ToPercentCoords(object):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, input_dict: dict):
        img = input_dict['img']
        boxes = input_dict.get('boxes', None)
        if boxes is None:
            return input_dict

        height, width, channels = img.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height
        input_dict['boxes'] = boxes
        return input_dict

class ResizeImgBoxes(object):
    def __init__(self, size, resize_boxes=True):
        self.size = size
        self.resize_boxes = resize_boxes

    def __call__(self, input_dict: dict):
        # Input boxes should be absolute pixel coordinates
        if isinstance(self.size, tuple):
            ih, iw = self.size[1], self.size[0]
        else:
            ih, iw = self.size, self.size
        img = input_dict['img']
        h, w = img.shape[:2]
        scale = min(iw/w, ih/h)
        nw, nh = int(scale*w), int(scale*h)
        img_resized = cv2.resize(img, (nw, nh))
        dw, dh = (iw-nw) // 2, (ih-nh) // 2
        img_padded = np.full(shape=[ih, iw, 3], fill_value=0)
        img_padded[dh:nh+dh, dw:dw+nw, :] = img_resized
        input_dict['img'] = img_padded
        input_dict['scale'] = scale
        input_dict['padding'] = [dw, dh]
        input_dict['rescale_size'] = [nw, nh]

        boxes = input_dict.get('boxes', None)
        if boxes is None or not self.resize_boxes:
            return input_dict

        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + dw
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + dh
        input_dict['boxes'] = boxes
        return input_dict

class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, input_dict: dict):
        if random.randint(2):
            img = input_dict['img']
            img[:, :, 1] *= random.uniform(self.lower, self.upper)
            input_dict['img'] = img
        return input_dict

class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, input_dict: dict):
        if random.randint(2):
            img = input_dict['img']
            img[:, :, 0] += random.uniform(-self.delta, self.delta)
            img[:, :, 0][img[:, :, 0] > 360.0] -= 360.0
            img[:, :, 0][img[:, :, 0] < 0.0] += 360.0
            input_dict['img'] = img
        return input_dict

class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, input_dict: dict):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)
            img = input_dict['img']
            img = shuffle(img)
            input_dict['img'] = img
        return input_dict

class ConvertColor(object):
    def __init__(self, current, transform):
        self.transform = transform
        self.current = current

    def __call__(self, input_dict: dict):
        img = input_dict['img']
        if self.current == 'BGR' and self.transform == 'HSV':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif self.current == 'RGB' and self.transform == 'HSV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif self.current == 'BGR' and self.transform == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif self.current == 'HSV' and self.transform == 'BGR':
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        elif self.current == 'HSV' and self.transform == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        input_dict['img'] = img
        return input_dict

class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, input_dict: dict):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            img = input_dict['img']
            img *= alpha
            input_dict['img'] = img
        return input_dict

class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, input_dict: dict):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            img = input_dict['img']
            img += delta
            input_dict['img'] = img
        return input_dict

class ToTensor(object):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, input_dict: dict):
        img = input_dict['img']
        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
        input_dict['img'] = img
        if 'padding' in input_dict:
            input_dict['padding'] = torch.tensor(input_dict['padding'])
        if 'rescale_size' in input_dict:
            input_dict['rescale_size'] = torch.tensor(input_dict['rescale_size'])
        return input_dict

class RandomSampleCrop(object):
    def __init__(self, *args, **kwargs):
        self.sample_options = (
            None,
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        )

    def __call__(self, input_dict: dict):
        img = input_dict['img']
        boxes = input_dict.get('boxes', None)
        labels = input_dict.get('labels', None)
        if boxes is not None and boxes.shape[0] == 0:
            return input_dict
        height, width, _ = img.shape
        while True:
            mode = self.sample_options[random.randint(0, len(self.sample_options))]
            if mode is None:
                return input_dict
            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')
            for _ in range(50):
                current_img = img
                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                if h / w < 0.5 or h / w > 2:
                    continue
                
                left = random.uniform(width - w)
                top = random.uniform(height - h)

                rect = np.array([int(left), int(top), int(left + w), int(top + h)])
                overlap = get_iou_numpy(boxes, rect)
                if overlap.max() < min_iou or overlap.max() > max_iou:
                    continue

                current_img = current_img[rect[1]:rect[3], rect[0]:rect[2], :]
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                mask = m1 * m2
                if not mask.any():
                    continue

                current_boxes = boxes[mask, :].copy()
                current_labels = labels[mask]
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                current_boxes[:, :2] -= rect[:2]
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                current_boxes[:, 2:] -= rect[:2]

                input_dict['img'] = current_img
                input_dict['boxes'] = current_boxes
                input_dict['labels'] = current_labels
                return input_dict

class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, input_dict: dict):
        if random.randint(2):
            return input_dict
        img = input_dict['img']
        boxes = input_dict.get('boxes', None)
        height, width, depth = img.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)
        expand_img = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=img.dtype
        )
        expand_img[:, :, :] = self.mean
        expand_img[int(top):int(top+height), int(left):int(left+width)] = img
        input_dict['img'] = expand_img

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))
        input_dict['boxes'] = boxes
        return input_dict

class RandomMirror(object):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, input_dict: dict):
        img = input_dict['img']
        boxes = input_dict.get('boxes', None)
        _, width, _ = img.shape
        if random.randint(2):
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
            input_dict['img'] = img[:, ::-1]
            input_dict['boxes'] = boxes
        return input_dict

class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, img):
        img = img[:, :, self.swaps]
        return img

class PhotometricDistort(object):
    def __init__(self, *args, **kwargs):
        self.pd = [
            RandomContrast(),
            ConvertColor(current="RGB", transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='RGB'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, input_dict: dict):
        input_dict = self.rand_brightness(input_dict)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        input_dict = distort(input_dict)
        return self.rand_light_noise(input_dict)