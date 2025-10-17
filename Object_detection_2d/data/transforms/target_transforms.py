import numpy as np
import torch

from Object_detection_2d.SSD.utils.box_utils import *
from Object_detection_2d.CenterNet.utils.hm_utils import *

class SSDTargetTransformOffset:
    def __init__(self, priors_cxcy, args):
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.center_variance = args['center_variance']
        self.size_variance = args['size_variance']
        self.iou_thres = args['iou_thres']

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)

        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes_xy, labels = assign_priors(gt_boxes, gt_labels, self.priors_xy, self.iou_thres)
        boxes_cxcy = xy_to_cxcy(boxes_xy)
        offsets = cxcy_to_offset(boxes_cxcy, self.priors_cxcy, self.center_variance, self.size_variance)
        return {"bboxes": offsets, "labels": labels}

class SSDTargetTransformCoord:
    def __init__(self, priors_cxcy, args):
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.center_variance = args['center_variance']
        self.size_variance = args['size_variance']
        self.iou_thres = args['iou_thres']

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)

        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes_xy, labels = assign_priors(gt_boxes, gt_labels, self.priors_xy, self.iou_thres)
        return {"bboxes": boxes_xy, "labels": labels}

class CenterNetTargetTransform:
    def __init__(self, args):
        self.img_size = args['img_size']
        self.class_num = args['class_num']
        self.max_objs = args['max_objs']

    def __call__(self, gt_boxes, gt_labels):
        output_size = self.img_size // 4
        hm = np.zeros((self.class_num, output_size, output_size), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        offsets = np.zeros((self.max_objs, 2), dtype=np.float32)

        ind = np.zeros((self.max_objs), dtype=np.int64)
        offsets_mask = np.zeros((self.max_objs), dtype=np.uint8)
        wh_concat = np.zeros((self.max_objs, self.class_num * 2), dtype=np.float32)
        mask_concat = np.zeros((self.max_objs, self.class_num * 2), dtype=np.uint8)

        for k in range(min(gt_boxes.shape[0], self.max_objs)):
            box = gt_boxes[k]
            cls_id = int(gt_labels[k])
            box = box / 4.0
            h, w = box[3] - box[1], box[2] - box[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))

                ct = np.array([(box[0]+box[2]) / 2, (box[1]+box[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)

                draw_umich_gaussian(hm[cls_id], ct_int, radius)

                wh[k] = 1.0 * w, 1.0 * h
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