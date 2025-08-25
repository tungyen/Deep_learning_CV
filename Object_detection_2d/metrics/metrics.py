from __future__ import division
import numpy as np
import itertools
import six
from collections import defaultdict

from Object_detection_2d.SSD.utils.box_utils import get_iou_numpy

def compute_object_detection_metrics(dataset, preds, iou_thres=0.5):
    pred_boxes_all = []
    pred_labels_all = []
    pred_scores_all = []
    gt_boxes_all = []
    gt_labels_all = []
    gt_difficulties_all = []

    for i in range(len(dataset)):
        img_id, annotation = dataset.get_annotation(i)
        gt_boxes, gt_labels, gt_difficulties = annotation
        gt_boxes_all.append(gt_boxes)
        gt_labels_all.append(gt_labels)
        gt_difficulties_all.append(gt_difficulties)

        img_info = dataset.get_img_info(i)
        pred = preds[i]
        pred = pred.resize((img_info['width'], img_info['height'])).numpy()
        boxes, labels, scores = pred['boxes'], pred['labels'], pred['scores']

        pred_boxes_all.append(boxes)
        pred_labels_all.append(labels)
        pred_scores_all.append(scores)
    
    precision, recall = compute_prec_rec(pred_boxes_all, pred_labels_all, pred_scores_all,
                                         gt_boxes_all, gt_labels_all, gt_difficulties_all, iou_thres)
    ap = compute_ap(precision, recall)
    return {'ap': ap, 'map': np.nanmean(ap)}


def compute_prec_rec(pred_boxes_all, pred_labels_all, pred_scores_all,
                     gt_boxes_all, gt_labels_all, gt_difficulties_all, iou_thres):
    pred_boxes_all = iter(pred_boxes_all)
    pred_labels_all = iter(pred_labels_all)
    pred_scores_all = iter(pred_scores_all)
    gt_boxes_all = iter(gt_boxes_all)
    gt_labels_all = iter(gt_labels_all)
    gt_difficulties_all = iter(gt_difficulties_all)

    n_pos = defaultdict(int)
    score = defaultdict(list)
    # For each class, match[c][i] = 1 for tp, 0 for fp, and -1 for ignore.
    match = defaultdict(list)

    for pred_boxes, pred_labels, pred_scores ,gt_boxes, gt_labels, gt_difficulties in \
            six.moves.zip(pred_boxes_all, pred_labels_all, pred_scores_all,
                          gt_boxes_all, gt_labels_all, gt_difficulties_all):
        for c in np.unique(np.concatenate((pred_labels, gt_labels)).astype(int)):
            pred_mask_class = pred_labels == c
            pred_boxes_class = pred_boxes[pred_mask_class]
            pred_scores_class = pred_scores[pred_mask_class]
            order = pred_scores_class.argsort()[::-1]
            pred_boxes_class = pred_boxes_class[order]
            pred_scores_class[order]

            gt_mask_class = gt_labels == c
            gt_boxes_class = gt_boxes[gt_mask_class]
            gt_difficulties_class = gt_difficulties[gt_mask_class]

            n_pos[c] += np.logical_not(gt_difficulties_class).sum()
            score[c].extend(pred_scores_class)

            if len(pred_boxes_class) == 0:
                continue
            if len(gt_boxes_class) == 0:
                match[c].extend((0, ) * pred_boxes_class.shape[0])
                continue

            pred_boxes_class = pred_boxes_class.copy()
            pred_boxes_class[:, 2:] += 1
            gt_boxes_class = gt_boxes_class.copy()
            gt_boxes_class[:, 2:] += 1

            iou = get_iou_numpy(pred_boxes_class, gt_boxes_class)
            gt_index = iou.argmax(axis=1)
            gt_index[iou.max(axis=1) < iou_thres] = -1
            del iou

            select = np.zeros(gt_boxes_class.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficulties_class[gt_idx]:
                        match[c].append(-1)
                    else:
                        if not select[gt_idx]:
                            match[c].append(1)
                        else:
                            match[c].append(0)
                    select[gt_idx] = True
                else:
                    match[c].append(0)

    for iter_ in (pred_boxes_all, pred_labels_all, pred_scores_all,
                  gt_boxes_all, gt_labels_all, gt_difficulties_all):
        if next(iter_, None) is not None:
            raise ValueError("Length of input does not match.")

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for c in n_pos.keys():
        score_class = np.array(score[c])
        match_class = np.array(match[c], dtype=np.int8)

        order = score_class.argsort()[::-1]
        match_class = match_class[order]

        tp = np.cumsum(match_class == 1)
        fp = np.cumsum(match_class == 0)
        prec[c] = tp / (tp + fp)
        if n_pos[c] > 0:
            rec[c] = tp / n_pos[c]
    return prec, rec

def compute_ap(prec, rec):
    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for c in six.moves.range(n_fg_class):
        if prec[c] is None or rec[c] is None:
            ap[c] = np.nan
            continue

        mpre = np.concatenate(([0], np.nan_to_num(prec[c]), [0]))
        mrec = np.concatenate(([0], rec[c], [1]))

        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap[c] = np.sum((mrec[i+1] - mrec[i]) * mpre[i+1])

    return ap
    
