import random
from torchvision.transforms import functional as F
import torch.nn as nn
import torch
import numpy as np
import math


def resizeImg(img, sizeMin, sizeMax):
    imgSize = torch.tensor(img.shape[-2:])
    imgSizeMin = float(torch.min(imgSize))
    imgSizeMax = float(torch.max(imgSize))
    s = imgSizeMin / sizeMin
    
    if imgSizeMax * s > sizeMax:
        s = sizeMax / imgSizeMax
        
    img = torch.nn.functional.interpolate(torch.unsqueeze(img, 0), 
                                          scale_factor=s,
                                          mode='bilinear',
                                          recompute_scale_factor=True,
                                          align_corners=False)[0]
    return img

def resizeBoundingBoxes(bboxes, originalSize, newSize):
    hRatio = newSize[0] / originalSize[0]
    wRatio = newSize[1] / originalSize[1]
    xMin, yMin, xMax, yMax = bboxes.unbind(1)
    xMin *= wRatio
    xMax *= wRatio
    yMin *= hRatio
    yMax *= hRatio
    return torch.stack((xMin, yMin, xMax, yMax), dim=1)
    

class resizeImageLabel(nn.Module):
    
    def __init__(self, sizeMin, sizeMax, imgMean, imgStd):
        super(resizeImageLabel, self).__init__()
        self.sizeMax = sizeMax
        self.sizeMin = sizeMin
        self.imgMean = imgMean
        self.imgStd = imgStd
        
    def normalized(self, img):
        dtype, device = img.dtype, img.device
        mu = torch.as_tensor(self.imgMean, dtype=dtype, device=device)
        std = torch.as_tensor(self.imgStd, dtype=dtype, device=device)
        return (img-mu[:, None, None]) / std[:, None, None]
    
    def resize(self, img, annotation):
        h, w = img.shape[-2:]
        img = resizeImg(img, float(self.sizeMin), float(self.sizeMax))
        
        if annotation is None:
            return img, annotation
        
        bboxes = annotation['bboxes']
        bboxes = resizeBoundingBoxes(bboxes, [h, w], img.shape[-2:])
        annotation['bboxes'] = bboxes
        return img, annotation
    
    def imgBatching(self, img, divide=32):
        shapeList = np.array([i.shape for i in img])
        maxSize = np.amax(shapeList, axis=0)
        stride = float(divide)
        maxSize[1] = int(math.ceil(float(maxSize[1]) / stride) * stride)
        maxSize[2] = int(math.ceil(float(maxSize[2]) / stride) * stride)
        batchShape = [len(img)] + maxSize
        batchImg = img[0].new_full(batchShape, 0)
        
        for i, bi in zip(img, batchImg):
            bi[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(i)

        return batchImg
    
    def turnBackImg(self, prediction, imgShape, originalShape):
        
        for i, (pred, s, s_ori) in enumerate(zip(prediction, imgShape, originalShape)):
            bboxes = pred['bboxes']
            bboxes = resizeBoundingBoxes(bboxes, s, s_ori)
            prediction[i] = bboxes
        return prediction
    
    def forward(self, imgs, annotations=None):
        
        for i in range(len(imgs)):
            img = imgs[i]
            annotation = annotations[i] if annotations is not None else None
            
            img = self.normalized(img)
            img, annotation = self.resize(img, annotation)
            imgs[i] = img
            if annotations is not None and annotation is not None:
                annotations[i] = annotation
                
        imgShape = [img.shape[-2:] for img in imgs]
        imgs = self.imgBatching(imgs)
        return imgs, imgShape, annotations
        


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            _, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["bboxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["bboxes"] = bbox
        return image, target
