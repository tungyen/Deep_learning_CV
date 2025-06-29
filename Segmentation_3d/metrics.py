import numpy as np

def compute_pcloud_seg_metrics(args, all_preds, all_labels):
    class_num = args.seg_class_num
        
    all_preds = all_preds.reshape(-1)
    all_labels = all_labels.reshape(-1)
    class_ious = []

    for cls in range(class_num):
        pred_mask = (all_preds == cls)
        target_mask = (all_labels == cls)

        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()

        if union == 0:
            iou = 0.0
        else:
            iou = intersection / union
        class_ious.append(iou)

    # Compute mean IoU for each class
    miou = np.mean(class_ious)
    return class_ious, miou

def compute_pcloud_cls_metrics(args, all_preds, all_labels):
    class_num = args.cls_class_num
    
    accuracy = (all_preds == all_labels).sum() / len(all_labels)

    precision_per_class = []
    recall_per_class = []

    for c in range(class_num):
        true_positives = ((all_preds == c) & (all_labels == c)).sum()
        predicted_positives = (all_preds == c).sum()
        actual_positives = (all_labels == c).sum()

        precision_c = true_positives / predicted_positives if predicted_positives > 0 else 0.0
        recall_c = true_positives / actual_positives if actual_positives > 0 else 0.0

        precision_per_class.append(precision_c)
        recall_per_class.append(recall_c)

    precision = sum(precision_per_class) / class_num
    recall = sum(recall_per_class) / class_num

    return accuracy, precision, recall