import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random

from Object_detection_2d.SSD.utils.box_utils import *

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
            if boxes is not None:
                boxes, labels = remove_empty_boxes(boxes, labels)
        return img, boxes, labels

class Lambda(object):
    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd
    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)

class ConvertFromInts(object):
    def __call__(self, img, boxes=None, labels=None):
        return img.astype(np.float32), boxes, labels

class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, img, boxes=None, labels=None):
        img = img.astype(np.float32)
        img -= self.mean
        return img.astype(np.float32), boxes, labels

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img, boxes=None, labels=None):
        img = img.astype(np.float32)
        if np.max(img) > 1.0:
            img /= 255.0
        img -= self.mean
        img /= self.std
        return img.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, img, boxes=None, labels=None):
        height, width, channels = img.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height
        return img, boxes, labels

class ToPercentCoords(object):
    def __call__(self, img, boxes=None, labels=None):
        height, width, channels = img.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height
        return img, boxes, labels

class ResizeImg(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, boxes=None, labels=None):
        if isinstance(self.size, tuple):
            img = cv2.resize(img, (self,size[0], self.size[1]))
        else:
            img = cv2.resize(img, (self.size, self.size))
        return img, boxes, labels

class ResizeImgBoxes(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, boxes=None, labels=None):
        # Input boxes should be absolute pixel coordinates
        if isinstance(self.size, tuple):
            ih, iw = self.size[1], self.size[0]
        else:
            ih, iw = self.size, self.size
        
        h, w = img.shape[:2]
        scale = min(iw/w, ih/h)
        nw, nh = int(scale*w), int(scale*h)
        img_resized = cv2.resize(img, (nw, nh))
        dw, dh = (iw-nw) // 2, (ih-nh) // 2
        img_padded = np.full(shape=[ih, iw, 3], fill_value=0)
        img_padded[dh:nh+dh, dw:dw+nw, :] = img_resized

        if boxes is None:
            return img_padded, boxes, labels

        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + dw
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + dh
        return img_padded, boxes, labels

class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, img, boxes=None, labels=None):
        if random.randint(2):
            img[:, :, 1] *= random.uniform(self.lower, self.upper)
        return img, boxes, labels

class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, img, boxes=None, labels=None):
        if random.randint(2):
            img[:, :, 0] += random.uniform(-self.delta, self.delta)
            img[:, :, 0][img[:, :, 0] > 360.0] -= 360.0
            img[:, :, 0][img[:, :, 0] < 0.0] += 360.0
        return img, boxes, labels

class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, img, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)
            img = shuffle(img)
        return img, boxes, labels

class ConvertColor(object):
    def __init__(self, current, transform):
        self.transform = transform
        self.current = current

    def __call__(self, img, boxes=None, labels=None):
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
        return img, boxes, labels

class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, img, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            img *= alpha
        return img, boxes, labels

class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, img, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            img += delta
        return img, boxes, labels

class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels

class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels

class RandomSampleCrop(object):
    def __init__(self):
        self.sample_options = (
            None,
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        )

    def __call__(self, img, boxes=None, labels=None):
        if boxes is not None and boxes.shape[0] == 0:
            return img, boxes, labels
        height, width, _ = img.shape
        while True:
            mode = self.sample_options[random.randint(0, len(self.sample_options))]
            if mode is None:
                return img, boxes, labels
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
                current_boxes[:, 2:] = np.maximum(current_boxes[:, 2:], rect[2:])
                current_boxes[:, 2:] -= rect[:2]
                return current_img, current_boxes, current_labels

class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, img, boxes, labels):
        if random.randint(2):
            return img, boxes, labels
        
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
        img = expand_img

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))
        return img, boxes, labels

class RandomMirror(object):
    def __call__(self, img, boxes, labels):
        _, width, _ = img.shape
        if random.randint(2):
            img = img[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return img, boxes, labels

class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, img):
        img = img[:, :, self.swaps]
        return img

class PhotometricDistort(object):
    def __init__(self):
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

    def __call__(self, img, boxes, labels):
        im = img.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)