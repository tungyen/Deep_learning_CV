import numpy as np
import torch
from tqdm import tqdm
import torch.distributed as dist
import torchvision.transforms.functional as F
from PIL import Image
from rich.table import Table
from rich.console import Console

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

    def update(self, preds, input_dict: dict):
        labels = input_dict["label"]

        if self.task != "img_seg":
            preds = preds.view(-1).to(torch.int64)
            labels = labels.view(-1).to(torch.int64)
        else:
            preds, labels = self.get_img_seg_data(preds, input_dict)

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
        f1_score = 2 * precision * recall / (precision + recall + self.eps)

        results = {
            'ious': ious,
            'mious': ious.mean(),
            'precision': precision,
            'mean_precision': precision.mean(),
            'recall': recall,
            'mean_recall': recall.mean(),
            'f1_score': f1_score,
            'mean_f1_score': f1_score.mean()
        }

        if self.task == "cls":
            self._print_table(results, show_iou=False)
            return results['mean_f1_score']

        elif self.task in ("semseg", "img_seg"):
            self._print_table(results, show_iou=True)
            return results['mious']

    def _print_table(self, results, show_iou=True):
        console = Console()
        table = Table(title="Validation Metrics", show_header=True, header_style="bold cyan", title_style="bold white")
        table.add_column("Class", style="bold white", justify="left")
        if show_iou:
            table.add_column("IoU", justify="center")
        table.add_column("Precision", justify="center")
        table.add_column("Recall",    justify="center")
        table.add_column("F1 Score",  justify="center")

        for cls in self.class_dict:
            if cls >= len(results['ious']):
                continue
            row = [self.class_dict[cls]]
            if show_iou:
                row.append(f"{results['ious'][cls]:.4f}")
            row += [
                f"{results['precision'][cls]:.4f}",
                f"{results['recall'][cls]:.4f}",
                f"{results['f1_score'][cls]:.4f}",
            ]
            table.add_row(*row)

        table.add_section()
        mean_row = ["[bold yellow]Mean[/bold yellow]"]
        if show_iou:
            mean_row.append(f"[bold yellow]{results['mious']:.4f}[/bold yellow]")
        mean_row += [
            f"[bold yellow]{results['mean_precision']:.4f}[/bold yellow]",
            f"[bold yellow]{results['mean_recall']:.4f}[/bold yellow]",
            f"[bold yellow]{results['mean_f1_score']:.4f}[/bold yellow]",
        ]
        table.add_row(*mean_row)
        console.print(table)

    def gather(self, local_rank):
        if is_main_process():
            tensor = self.confusion_matrix.to(torch.device(f"cuda:{local_rank}"))
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            self.confusion_matrix = tensor

    def reset(self):
        self.confusion_matrix.zero_()
        self.confusion_matrix = self.confusion_matrix.cpu()

    
    def get_img_seg_data(self, output, input_dict: dict):
        labels = input_dict['label']
        bs = labels.shape[0]
        ori_sizes = input_dict['original_size']

        preds = []
        targets = []

        for i in range(bs):
            pred = output[i]
            label = labels[i]
            ori_size = ori_sizes[i]

            if "padding" in input_dict and "rescale_size" in input_dict:
                paddings = input_dict['padding']
                rescale_sizes = input_dict['rescale_size']
                dw, dh = paddings[i]
                nw, nh = rescale_sizes[i]
                ori_size = ori_sizes[i]

                pred = pred[dh:nh+dh, dw:dw+nw]
                label = label[dh:nh+dh, dw:dw+nw]
            pred = F.resize(pred.unsqueeze(0), ori_size, Image.NEAREST).view(-1).to(torch.int64)
            label = F.resize(label.unsqueeze(0), ori_size, Image.NEAREST).view(-1).to(torch.int64)
            preds.append(pred)
            targets.append(label)

        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)
        return preds, targets

