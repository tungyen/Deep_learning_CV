import torch
import numpy as np
import math

def get_intersect_numpy(box1, box2):
    right_coord = np.minimum(box1[:, 2:], box2[2:])
    left_coord = np.maximum(box1[:, :2], box2[:2])
    inter = np.clip((right_coord - left_coord), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]

def get_iou_numpy(box1, box2):
    """
    Args:
        box1: Multiple bounding boxes with shape (num_boxes, 4)
        box2: Single bounding box with shape (4)
    """
    inter = get_intersect_numpy(box1, box2)
    area_1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area_2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area_1 + area_2 - inter
    return inter / union

def get_iou_numpy_multiboxes(box1, box2):
    """
    Args:
        box1: Multiple bounding boxes with shape (num_boxes_1, 4)
        box2: Multiple bounding boxes with shape (num_boxes_2, 4)
    """
    right_coord = np.minimum(box1[:, None, 2:], box2[:, 2:])
    left_coord = np.maximum(box1[:, None, :2], box2[:, :2])
    inter = np.prod(right_coord - left_coord, axis=2) * (left_coord < right_coord).all(axis=2)
    area_1 = np.prod(box1[:, 2:] - box1[:, :2], axis=1)
    area_2 = np.prod(box2[:, 2:] - box2[:, :2], axis=1)
    union = area_1[:, None] + area_2 - inter
    return inter / union

def remove_empty_boxes(boxes, labels):
    del_boxes = []
    for idx, box in enumerate(boxes):
        if box[0] == box[2] or box[1] == box[3]:
            del_boxes.append(idx)
    return np.delete(boxes, del_boxes, 0), np.delete(labels, del_boxes)

def cxcy_to_xy(boxes_cxcy):
    return torch.cat([boxes_cxcy[..., :2] - boxes_cxcy[..., 2:] / 2,
                      boxes_cxcy[..., :2] + boxes_cxcy[..., 2:] / 2], boxes_cxcy.dim() - 1)

def xy_to_cxcy(boxes_xy):
    return torch.cat([(boxes_xy[..., :2] + boxes_xy[..., 2:]) / 2,
                       boxes_xy[..., 2:] - boxes_xy[..., :2]], boxes_xy.dim() - 1)

def offset_to_cxcy(offset, priors, center_variance, size_variance):
    if priors.dim() + 1 == offset.dim():
        priors = priors.unsqueeze(0)
    return torch.cat([
        offset[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        torch.exp(offset[..., 2:] * size_variance) * priors[..., 2:]
    ], dim=offset.dim() - 1)

def cxcy_to_offset(boxes_cxcy, priors_cxcy, center_variance, size_variance):
    if priors_cxcy.dim() + 1 == boxes_cxcy.dim():
        priors_cxcy = priors_cxcy.unsqueeze(0)
    return torch.cat([
        (boxes_cxcy[..., :2] - priors_cxcy[..., :2]) / priors_cxcy[..., 2:] / center_variance,
        torch.log(boxes_cxcy[..., 2:] / priors_cxcy[..., 2:]) / size_variance
    ], dim=boxes_cxcy.dim() - 1)

def get_area(left_top, right_bottom):
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]

def get_iou(boxes_1, boxes_2, eps=1e-5):
    overlap_left_top = torch.max(boxes_1[..., :2], boxes_2[..., :2])
    overlap_right_bottom = torch.min(boxes_1[..., 2:], boxes_2[..., 2:])

    intersect = get_area(overlap_left_top, overlap_right_bottom)
    area_1 = get_area(boxes_1[..., :2], boxes_1[..., 2:])
    area_2 = get_area(boxes_2[..., :2], boxes_2[..., 2:])
    return intersect / (area_1 + area_2 - intersect + eps)

def assign_priors(gt_boxes, gt_labels, priors_xy, iou_thres):
    ious = get_iou(gt_boxes.unsqueeze(0), priors_xy.unsqueeze(1))
    best_gt_per_prior, best_gt_per_prior_idx = ious.max(1)
    best_prior_per_gt, best_prior_per_gt_idx = ious.max(0)

    for gt_idx, prior_idx in enumerate(best_prior_per_gt_idx):
        best_gt_per_prior_idx[prior_idx] = gt_idx
    best_gt_per_prior.index_fill_(0, best_prior_per_gt_idx, 2)
    labels = gt_labels[best_gt_per_prior_idx]
    labels[best_gt_per_prior < iou_thres] = 0
    boxes = gt_boxes[best_gt_per_prior_idx]
    return boxes, labels

def hard_negative_mining(loss, labels, neg_pos_ratio):
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask
