import numpy as np
import torch

from Object_detection_2d.SSD.utils.box_utils import *

class SSDTargetTransformOffset:
    def __init__(self, priors_cxcy, center_variance, size_variance, iou_thres):
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_thres = iou_thres

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)

        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes_xy, labels = assign_priors(gt_boxes, gt_labels, self.priors_xy, self.iou_thres)
        boxes_cxcy = xy_to_cxcy(boxes_xy)
        offsets = cxcy_to_offset(boxes_cxcy, self.priors_cxcy, self.center_variance, self.size_variance)
        return offsets, labels

class SSDTargetTransformCoord:
    def __init__(self, priors_cxcy, center_variance, size_variance, iou_thres):
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_thres = iou_thres

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)

        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes_xy, labels = assign_priors(gt_boxes, gt_labels, self.priors_xy, self.iou_thres)
        return boxes_xy, labels