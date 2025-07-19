import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from Object_detection_2d.utils import xy_to_cxcy, cxcy_to_xy, cxcy_to_offset, find_boxes_overlap

class BaseIouLoss(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps
        
    def forward(self, pred_boxes, gt_boxes):
        loss = self.get_loss(pred_boxes, gt_boxes)
        return torch.mean(loss)
    
    def get_loss(self, pred_boxes, gt_boxes):
        raise NotImplementedError("Subclasses must implement `get_loss()`")
    
    def get_boxes_area(self, boxes):
        w = (boxes[:, 2] - boxes[:, 0] + 1.0).clamp(min=0)
        h = (boxes[:, 3] - boxes[:, 1] + 1.0).clamp(min=0)
        return w * h
    
    def get_ious(self, pred_boxes, gt_boxes):
        x1 = torch.max(pred_boxes[:, 0], gt_boxes[:, 0])
        y1 = torch.max(pred_boxes[:, 1], gt_boxes[:, 1])
        x2 = torch.min(pred_boxes[:, 2], gt_boxes[:, 2])
        y2 = torch.min(pred_boxes[:, 3], gt_boxes[:, 3])
        
        w = (x2 - x1 + 1.0).clamp(0.0)
        h = (y2 - y1 + 1.0).clamp(0.0)
        
        intersection = w * h
        pred_areas = (pred_boxes[:, 2] - pred_boxes[:, 0] + 1.0) * (pred_boxes[:, 3] - pred_boxes[:, 1] + 1.0)
        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0)
        unions = pred_areas + gt_areas - intersection
        ious = intersection / (unions)
        return ious, intersection, unions
    
    def get_internal_diag_dist(self, pred_boxes, gt_boxes):
        pred_cx = (pred_boxes[:, 2] + pred_boxes[:, 0]) / 2
        pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
        
        gt_cx = (gt_boxes[:, 2] + gt_boxes[:, 0]) / 2
        gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
        internal_diag_distance = (pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2
        return internal_diag_distance
    
    def get_enclose(self, pred_boxes, gt_boxes):
        ex1 = torch.min(pred_boxes[:, 0], gt_boxes[:, 0])
        ey1 = torch.min(pred_boxes[:, 1], gt_boxes[:, 1])
        ex2 = torch.max(pred_boxes[:, 2], gt_boxes[:, 2])
        ey2 = torch.max(pred_boxes[:, 3], gt_boxes[:, 3])
        
        ew = (ex2 - ex1 + 1.0).clamp(0.0)
        eh = (ey2 - ey1 + 1.0).clamp(0.0)
        enclose = ew * eh + self.eps
        outside_diag_dist = (ex1 - ex2) ** 2 + (ey1 - ey2) ** 2
        return enclose, outside_diag_dist

class IouLoss(BaseIouLoss):
    def get_loss(self, pred_boxes, gt_boxes):
        ious, _, _ = self.get_ious(pred_boxes, gt_boxes)
        ious = ious.clamp(min=self.eps)
        loss = -ious.log()
        return torch.mean(loss)

class GiouLoss(BaseIouLoss):
    def get_loss(self, pred_boxes, gt_boxes):
        ious, _, unions = self.get_ious(pred_boxes, gt_boxes)
        enclose, _ = self.get_enclose(pred_boxes, gt_boxes)
        gious = ious - (enclose - unions) / enclose
        loss = 1 - gious
        return torch.mean(loss)
    
class DiouLoss(BaseIouLoss): 
    def get_loss(self, pred_boxes, gt_boxes):
        dious, _ = self.get_dious(pred_boxes, gt_boxes)
        loss = 1 - dious
        return torch.mean(loss)
    
    def get_dious(self, pred_boxes, gt_boxes):
        ious, _, _ = self.get_ious(pred_boxes, gt_boxes)
        internal_diag_distance = self.get_internal_diag_dist(pred_boxes, gt_boxes)
        
        _, outside_diag_dist = self.get_enclose(pred_boxes, gt_boxes)
        dious = ious - internal_diag_distance / outside_diag_dist
        dious = torch.clamp(dious, min=-1.0, max=1.0)
        return dious, ious

class CiouLoss(DiouLoss):
    def get_loss(self, pred_boxes, gt_boxes):
        dious, ious = self.get_dious(pred_boxes, gt_boxes)
        pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
        pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
        gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
        gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
        
        v = torch.pow((torch.atan(gt_w / gt_h) - torch.atan(pred_w, pred_h)), 2) * (4 / (math.pi ** 2))
        alpha = v / (1-ious + v)
        cious = dious - alpha * v
        cious = torch.clamp(cious, min=-1.0, max=1.0)
        loss = 1 - cious
        return torch.mean(loss)

class MultiBoxesLoss(nn.Module):
    def __init__(self, prior_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.0):
        super().__init__()
        self.prior_cxcy = prior_cxcy
        self.prior_num = prior_cxcy.shape[0]
        self.prior_xy = cxcy_to_xy(prior_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        
        self.smooth_l1 = nn.SmoothL1Loss()
        self.ce = nn.CrossEntropyLoss()
        
    def forward(self, pred_boxes, pred_logits, boxes, labels):
        batch_size = pred_boxes.shape[0]
        class_num = pred_logits.shape[1]
        device = pred_boxes.device
        assert self.prior_num == pred_boxes.shape[2] == pred_logits.shape[2]
        
        true_boxes = torch.zeros((batch_size, self.prior_xy, 4), dtype=torch.float).to(device)
        true_classes = torch.zeros((batch_size, self.prior_num), dtype=torch.long).to(device)
        
        for i in range(batch_size):
            gt_num = boxes[i].shape[0]
            overlap = find_boxes_overlap(boxes[i], self.prior_xy) # (gt_num, self.prior_num)
            overlap_per_prior, overlap_idx_per_prior = overlap.max(dim=0)
            _, overlap_idx_per_gt = overlap.max(dim=1)
            
            overlap_idx_per_prior[overlap_idx_per_gt] = torch.arange(gt_num).to(device)
            overlap_per_prior[overlap_idx_per_gt] = 1.0
            
            labels_per_prior = labels[i][overlap_idx_per_prior]
            labels_per_prior[overlap_per_prior < self.threshold] = 0.0
            
            true_classes[i] = labels_per_prior
            true_boxes[i] = cxcy_to_offset(xy_to_cxcy(boxes[i][overlap_idx_per_prior]), self.prior_cxcy)
            
        positive_priors = true_classes != 0
        boxes_loss = self.smooth_l1(pred_boxes[positive_priors], true_boxes[positive_priors])
        
        positive_num = positive_num.sum(dim=1)
        hard_negative_num = positive_num * self.neg_pos_ratio
        
        cls_loss = self.ce(pred_logits.view(-1, class_num), true_classes.view(-1))
        cls_loss = cls_loss.view(batch_size, self.prior_num)
        cls_loss_pos = cls_loss[positive_priors]
        
        cls_loss_neg = cls_loss.clone()
        cls_loss_neg[positive_priors] = 0.0
        cls_loss_neg, _ = torch.sort(cls_loss_neg, dim=1, descending=True)
        hardness_ranks = torch.LongTensor(range(self.prior_num)).unsqueeze(0).expand_as(cls_loss_neg).to(device)
        hard_negatives = hardness_ranks < hard_negative_num.unsqueeze(1)
        cls_loss_neg_hard = cls_loss_neg[hard_negatives]
        
        cls_loss = (cls_loss_neg_hard.sum() + cls_loss_pos.sum()) / positive_num.sum().float()
        return cls_loss + self.alpha * boxes_loss
    
    
def get_loss(args):
    if args.loss_func == 'CeSmoothL1':
        return 
    else:
        raise ValueError(f'Unknown loss function {args.loss_func}.')