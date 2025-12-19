import numpy as np
import torch
from tqdm import tqdm
import torch.distributed as dist

from core.utils import is_main_process

class ConfusionMatrix:
    def __init__(self, class_dict, class_num, dataset=None, task="seg", ignore_index=None, eps=1e-7, device='cpu'):
        self.class_num = class_num
        self.task = task
        self.ignore_index = ignore_index
        self.eps = eps
        self.device = device
        self.class_dict = class_dict
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

        results = {
            'ious': ious,
            'mious': ious.mean(),
            'precision': precision,
            'mean_precision': precision.mean(),
            'recall': recall,
            'mean_recall': recall.mean(),
        }

        if self.task == "cls":
            precision = results['mean_precision']
            recall = results['mean_recall']
            print("Validation Precision ===> {:.4f}".format(precision))
            print("Validation Recall ===> {:.4f}".format(recall))
            return precision

        elif self.task == "semseg":
            ious = results['ious']
            mious = results['mious']
            print("Validation mIoU ===> {:.4f}".format(mious))
            for cls in self.class_dict:
                if cls >= len(ious):
                    continue
                print("{} IoU: {:.4f}".format(self.class_dict[cls], ious[cls]))
            return mious

    def gather(self, local_rank):
        if is_main_process():
            tensor = self.confusion_matrix.to(torch.device(f"cuda:{local_rank}"))
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            self.confusion_matrix = tensor

    def reset(self):
        self.confusion_matrix.zero_()
        self.confusion_matrix = self.confusion_matrix.cpu()