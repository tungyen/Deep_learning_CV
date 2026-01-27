import numpy as np
import torch

from core.utils.box_utils import *
from core.utils.hm_utils import *

class Compose_target(object):
    def __init__(self, target_transforms):
        self.target_transforms = target_transforms
        
    def __call__(self, input_dict: dict, device=None):
        for t in self.target_transforms:
            input_dict = t(input_dict, device=device)
        return input_dict

class SSDTargetTransformOffset:
    def __init__(self, priors_cxcy, center_variance=0.1, size_variance=0.2,
                 iou_thres=0.5):
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_thres = iou_thres

    def __call__(self, input_dict: dict, device=None):
        gt_boxes = input_dict['boxes']
        gt_labels = input_dict['labels']
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)

        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes_xy, labels = assign_priors(gt_boxes, gt_labels, self.priors_xy, self.iou_thres)
        boxes_cxcy = xy_to_cxcy(boxes_xy)
        offsets = cxcy_to_offset(boxes_cxcy, self.priors_cxcy, self.center_variance, self.size_variance)

        input_dict['boxes'] = offsets
        input_dict['labels'] = labels
        return input_dict

class SSDTargetTransformCoord:
    def __init__(self, priors_cxcy, center_variance=0.1, size_variance=0.2,
                 iou_thres=0.5):
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_thres = iou_thres

    def __call__(self, input_dict: dict, device=None):
        gt_boxes = input_dict['boxes']
        gt_labels = input_dict['labels']
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)

        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes_xy, labels = assign_priors(gt_boxes, gt_labels, self.priors_xy, self.iou_thres)
        input_dict['boxes'] = boxes_xy
        input_dict['labels'] = labels
        return input_dict

class CenterNetTargetTransform:
    def __init__(self, img_size=300, class_num=21, max_objs=100):
        self.img_size = img_size
        self.class_num = class_num
        self.max_objs = max_objs

    def __call__(self, input_dict: dict, device=None):
        gt_boxes = input_dict['boxes']
        gt_labels = input_dict['labels']
        output_size = self.img_size // 4
        hm = torch.zeros((self.class_num, output_size, output_size), dtype=torch.float32, device=device)
        wh = torch.zeros((self.max_objs, 2), dtype=torch.float32, device=device)
        offsets = torch.zeros((self.max_objs, 2), dtype=torch.float32, device=device)

        ind = torch.zeros((self.max_objs), dtype=torch.long, device=device)
        offsets_mask = torch.zeros((self.max_objs), dtype=torch.uint8, device=device)

        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes).to(device)
        gt_boxes = gt_boxes.to(torch.float32) / 4.0
        gt_boxes[:, [0, 2]] = torch.clip(gt_boxes[:, [0, 2]], min=0, max=output_size-1)
        gt_boxes[:, [1, 3]] = torch.clip(gt_boxes[:, [1, 3]], min=0, max=output_size-1)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels).to(device)
        for k in range(min(gt_boxes.shape[0], self.max_objs)):
            box = gt_boxes[k]
            cls_id = int(gt_labels[k])
            h, w = box[3] - box[1], box[2] - box[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))

                ct = torch.tensor([(box[0]+box[2]) / 2, (box[1]+box[3]) / 2], device=device)
                ct_int = ct.to(torch.int32)

                draw_umich_gaussian(hm[cls_id], ct_int, radius)

                wh[k] = torch.tensor([1.0 * w, 1.0 * h], device=device)
                ind[k] = ct_int[1] * output_size + ct_int[0]
                offsets[k] = ct - ct_int
                offsets_mask[k] = 1

        input_dict['hm'] = hm
        input_dict['wh'] = wh
        input_dict['offsets'] = offsets
        input_dict['ind'] = ind
        input_dict['offsets_mask'] = offsets_mask
        input_dict.pop('boxes')
        input_dict.pop('labels')
        return input_dict