import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from Object_detection_2d.SSD.utils.box_utils import *

class MultiBoxesLoss(nn.Module):
    def __init__(self, neg_pos_ratio):
        super().__init__()
        self.neg_pos_ratio = neg_pos_ratio
        
    def forward(self, pred_boxes, pred_logits, gt_boxes, gt_labels):
        loss_dict = {}
        batch_size = pred_boxes.shape[0]
        class_num = pred_logits.shape[2]
        device = pred_boxes.device
        
        with torch.no_grad():
            loss = -F.log_softmax(pred_logits, dim=2)[:, :, 0]
            mask = hard_negative_mining(loss, gt_labels, self.neg_pos_ratio)
        
        pred_logits = pred_logits[mask, :]
        cls_loss = F.cross_entropy(pred_logits.view(-1, class_num), gt_labels[mask], reduction='sum')

        pos_mask = gt_labels > 0
        pred_boxes = pred_boxes[pos_mask, :].view(-1, 4)
        gt_boxes = gt_boxes[pos_mask, :].view(-1, 4)
        boxes_loss = F.smooth_l1_loss(pred_boxes, gt_boxes, reduction='sum')
        num_pos = gt_boxes.shape[0]

        loss_dict['boxes_loss'] = boxes_loss / num_pos
        loss_dict['cls_loss'] = cls_loss / num_pos
        loss_dict['loss'] = loss_dict['boxes_loss'] + loss_dict['cls_loss']
        return loss_dict
