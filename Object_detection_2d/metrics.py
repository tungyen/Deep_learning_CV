import numpy as np

from Object_detection_2d.utils import find_boxes_overlap

def compute_object_detection_mAP(args, pred_boxes, pred_labels, pred_scores, true_boxes, true_labels, true_difficulties):
    assert len(pred_boxes) == len(pred_labels) == len(pred_scores) == len(true_boxes) == len(true_labels) == len(true_difficulties)
    class_num = args.class_num
    device = pred_boxes.device
    iou_threshold = args.iou_threshold

    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.tensor(true_images, dtype=torch.long, device=device)
    true_boxes = torch.cat(true_boxes, dim=0)
    true_labels = torch.cat(true_labels, dim=0)
    true_difficulties = torch.cat(true_difficulties, dim=0)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0) == true_difficulties.size(0)

    pred_images = list()
    for i in range(len(pred_labels)):
        pred_images.extend([i] * pred_labels[i].size(0))
    pred_images = torch.tensor(pred_images, dtype=torch.long, device=device)
    pred_boxes = torch.cat(pred_boxes, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    pred_scores = torch.cat(pred_scores, dim=0)

    assert pred_images.size(0) == pred_boxes.size(0) == pred_labels.size(0) == pred_scores.size(0)

    APs = torch.zeros((class_num - 1, ), dtype=torch.float)
    for c in range(1, class_num):
        true_class_images = true_images[true_labels == c]
        true_class_boxes = true_boxes[true_labels == c]
        true_class_difficulties = true_difficulties[true_labels == c]
        easy_class_objects_num = (1 - true_class_difficulties).sum().item()

        true_class_boxes_detected = torch.zeros(true_class_boxes.size(0), dtype=torch.uint8, device=device)

        pred_class_images = pred_images[pred_labels == c]
        pred_class_boxes = pred_boxes[pred_labels == c]
        pred_class_scores = pred_scores[pred_labels == c]
        class_detections_num = pred_class_boxes.size(0)
        if class_detections_num == 0:
            continue

        pred_class_scores, indices = torch.sort(pred_class_scores, descending=True)
        pred_class_images = pred_class_images[indices]
        pred_class_boxes = pred_class_boxes[indices]

        TP = torch.zeros((class_detections_num, ), dtype=torch.float, device=device)
        FP = torch.zeros(c(lass_detections_num, ), dtype=torch.float, device=device)
        for i in range(class_detections_num):
            current_pred_box = pred_class_boxes[i].unsqueeze(0)
            current_image = pred_class_images[i]

            object_boxes = true_class_boxes[true_class_images == current_image]
            object_difficulties = true_class_difficulties[true_class_images == current_image]
            if object_boxes.size(0) == 0:
                FP[i] = 1.0
                continue

            overlaps = find_boxes_overlap(current_pred_box, object_boxes)
            max_overlap, max_idx = torch.max(overlaps, dim=0)
            original_idx = torch.tensor(range(true_class_boxes.size(0)))[true_class_images == current_image][max_idx]

            if max_overlap >= iou_threshold:
                if object_difficulties[max_idx] == 0:
                    if true_class_boxes_detected[original_idx] == 0:
                        TP[i] = 1.0
                        true_class_boxes_detected[original_idx] = 1
                    else:
                        FP[i] = 1.0
            else:
                FP[i] = 1.0

        cum_tp = torch.cumsum(TP, dim=0)
        cum_fp = torch.cumsum(FP, dim=0)
        cum_precision = cum_tp / (cum_tp + cum_fp + 1e-10)
        cum_recall = cum_tp / (easy_class_objects_num + 1e-10)

        recall_thresholds = torch.arange(start=0, end=1.1, step=0.1).tolist()
        precision = torch.zeros(len(recall_thresholds), dtype=torch.float, device=device)
        for i, thres in enumerate(recall_thresholds):
            recalls_above_thres = cum_recall >= thres
            if recalls_above_thres.any():
                precision[i] = cum_precision[recalls_above_thres].max()
            else:
                precision[i] = 0.0
        APs[c-1] = precision.mean()
    mAP = APs.mean().item()
    return mAP, APs