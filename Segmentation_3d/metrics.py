import numpy as np
import torch
from tqdm import tqdm

def compute_pcloud_partseg_metrics(all_preds, all_labels, class_dict):
    instance2parts, parts2instance, _ = class_dict
    instance_ious = {cls: [] for cls in instance2parts.keys()}

    for preds, labels in zip(all_preds, all_labels):
        labels = labels.numpy()
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

class ConfusionMatrix:
    def __init__(self, class_num, ignore_index=None, eps=1e-7, device='cpu'):
        self.class_num = class_num
        self.ignore_index = ignore_index
        self.eps = eps
        self.device = device
        self.confusion_matrix = torch.zeros((class_num, class_num), dtype=torch.int64, device=device)

    def update(self, preds, labels):
        preds = preds.view(-1).to(torch.int64)
        labels = labels.view(-1).to(torch.int64)

        if self.ignore_index is not None:
            mask = labels != self.ignore_index
            preds = preds[mask]
            labels = labels[mask]

        indexes = self.class_num * labels + preds
        cm = torch.bincount(indexes, minlength=self.class_num ** 2)
        cm = cm.reshape(self.class_num, self.class_num)
        self.confusion_matrix += cm

    def compute_metrics(self):
        TP = self.confusion_matrix.diag()
        FP = self.confusion_matrix.sum(0) - TP
        FN = self.confusion_matrix.sum(1) - TP
        ious = TP.float() / (TP + FP + FN + self.eps)
        precision = TP.float() / (TP + FP + self.eps)
        recall = TP.float() / (TP + FN + self.eps)

        return {
            'ious': ious,
            'mious': ious.mean(),
            'precision': precision,
            'mean_precision': precision.mean(),
            'recall': recall,
            'mean_recall': recall.mean(),
        }
    
    def reset(self):
        self.confusion_matrix.zero_()