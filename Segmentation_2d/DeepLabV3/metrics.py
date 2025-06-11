import numpy as np

def compute_image_seg_metrics(args, all_preds, all_labels):
    class_num = args.class_num
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