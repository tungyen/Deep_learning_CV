import numpy as np

def compute_pcloud_partseg_metrics(all_preds, all_labels, class_dict):
    instance2parts, parts2instance, _ = class_dict
    instance_ious = {cls: [] for cls in instance2parts.keys()}

    for preds, labels in zip(all_preds, all_labels):
        labels = labels.numpy()
        preds = preds.numpy()
        batch_size = preds.shape[0]
        for i in range(batch_size):
            pred = preds[i, :]
            label = labels[i, :]
            cls = parts2instance[label[0]]
            part_ious = [0.0 for _ in range(len(instance2parts[cls]))]
            for l in instance2parts[cls]:
                if (np.sum(label == l) == 0) and (np.sum(pred == l) == 0):
                    part_ious[l-instance2parts[cls][0]] = 1.0
                else:
                    part_ious[l-instance2parts[cls][0]] = np.sum((label == l) & (pred == l)) / float(np.sum((label == l) | (pred == l)))
            instance_ious[cls].append(np.mean(part_ious))
            
    all_instance_ious = []
    for cls in instance_ious.keys():
        for iou in instance_ious[cls]:
            all_instance_ious.append(iou)
        instance_ious[cls] = np.mean(instance_ious[cls])
    class_mious = np.mean(list(instance_ious.values()))
    instance_mious = np.mean(all_instance_ious)
    return instance_ious, instance_mious, class_mious