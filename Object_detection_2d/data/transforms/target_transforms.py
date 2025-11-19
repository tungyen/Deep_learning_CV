import numpy as np
import torch

from Object_detection_2d.SSD.utils.box_utils import *
from Object_detection_2d.CenterNet.utils.hm_utils import *

class Compose_target(object):
    def __init__(self, target_transforms):
        self.target_transforms = target_transforms
        
    def __call__(self, gt_boxes, gt_labels, device=None):
        for t in self.target_transforms:
            result_dict = t(gt_boxes=gt_boxes, gt_labels=gt_labels, device=device)
        return result_dict

class SSDTargetTransformOffset:
    def __init__(self, priors_cxcy, center_variance=0.1, size_variance=0.2,
                 iou_thres=0.5):
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_thres = iou_thres

    def __call__(self, gt_boxes, gt_labels, device=None):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)

        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes_xy, labels = assign_priors(gt_boxes, gt_labels, self.priors_xy, self.iou_thres)
        boxes_cxcy = xy_to_cxcy(boxes_xy)
        offsets = cxcy_to_offset(boxes_cxcy, self.priors_cxcy, self.center_variance, self.size_variance)
        return {"bboxes": offsets, "labels": labels}

class SSDTargetTransformCoord:
    def __init__(self, priors_cxcy, center_variance=0.1, size_variance=0.2,
                 iou_thres=0.5):
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_thres = iou_thres

    def __call__(self, gt_boxes, gt_labels, device=None):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)

        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes_xy, labels = assign_priors(gt_boxes, gt_labels, self.priors_xy, self.iou_thres)
        return {"bboxes": boxes_xy, "labels": labels}

class CenterNetTargetTransform:
    def __init__(self, img_size=300, class_num=21, max_objs=100):
        self.img_size = img_size
        self.class_num = class_num
        self.max_objs = max_objs

    def __call__(self, gt_boxes, gt_labels, device=None):
        output_size = self.img_size // 4
        hm = torch.zeros((self.class_num, output_size, output_size), dtype=torch.float32, device=device)
        wh = torch.zeros((self.max_objs, 2), dtype=torch.float32, device=device)
        offsets = torch.zeros((self.max_objs, 2), dtype=torch.float32, device=device)

        ind = torch.zeros((self.max_objs), dtype=torch.long, device=device)
        offsets_mask = torch.zeros((self.max_objs), dtype=torch.uint8, device=device)
        wh_concat = torch.zeros((self.max_objs, self.class_num * 2), dtype=torch.float32, device=device)
        mask_concat = torch.zeros((self.max_objs, self.class_num * 2), dtype=torch.uint8, device=device)

        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes).to(device)
        gt_boxes = gt_boxes / 4.0

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

                wh[k] = torch.tensor([w, h], device=device)
                ind[k] = ct_int[1] * output_size + ct_int[0]
                offsets[k] = ct - ct_int
                offsets_mask[k] = 1

                wh_concat[k, cls_id*2:cls_id*2+2] = wh[k]
                mask_concat[k, cls_id*2:cls_id*2+2] = 1

        result = {
            'hm': torch.from_numpy(hm),
            'wh': torch.from_numpy(wh),
            'offsets': torch.from_numpy(offsets),
            'ind': torch.from_numpy(ind),
            'offsets_mask': torch.from_numpy(offsets_mask)
        }
        return result