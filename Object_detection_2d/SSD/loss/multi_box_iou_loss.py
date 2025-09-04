import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from Object_detection_2d.SSD.utils.box_utils import *
from Object_detection_2d.SSD.loss.iou_loss import *

class MultiBoxesIouLoss(nn.Module):
    def __init__(self, priors_xy, neg_pos_ratio, box_loss_weight, center_variance, size_variance):
        super().__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.box_loss_weight = box_loss_weight
        self.iou_loss = CiouLoss(eps=1e-7)
        self.priors_xy = priors_xy
        self.center_variance = center_variance
        self.size_variance = size_variance
        
    def forward(self, pred_boxes, pred_logits, gt_boxes, gt_labels):
        loss_dict = {}
        batch_size = pred_boxes.shape[0]
        class_num = pred_logits.shape[2]
        device = pred_boxes.device
        self.priors_xy = self.priors_xy.to(device)

        boxes_cxcy = offset_to_cxcy(pred_boxes, self.priors_xy, self.center_variance, self.size_variance)
        pred_boxes = cxcy_to_xy(boxes_cxcy)

        with torch.no_grad():
            loss = -F.log_softmax(pred_logits, dim=2)[:, :, 0]
            mask = hard_negative_mining(loss, gt_labels, self.neg_pos_ratio)
        
        pred_logits = pred_logits[mask, :]
        cls_loss = F.cross_entropy(pred_logits.view(-1, class_num), gt_labels[mask], reduction='sum')

        pos_mask = gt_labels > 0
        pred_boxes = pred_boxes[pos_mask, :].view(-1, 4)
        gt_boxes = gt_boxes[pos_mask, :].view(-1, 4)
        boxes_loss = self.iou_loss(pred_boxes, gt_boxes)
        num_pos = gt_boxes.shape[0]

        loss_dict['boxes_loss'] = self.box_loss_weight * boxes_loss / num_pos
        loss_dict['cls_loss'] = cls_loss / num_pos
        loss_dict['loss'] = loss_dict['boxes_loss'] + loss_dict['cls_loss']
        return loss_dict
