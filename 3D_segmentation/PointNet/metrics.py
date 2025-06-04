import torch
from tqdm import tqdm
import numpy as np

def compute_pcloud_seg_metrics(args, val_dataloader, model):
    class_num = args.class_num
    iou_list = [[] for _ in range(class_num)]
        
    with torch.no_grad():
        for pcloud, label in tqdm(val_dataloader):
            output = model(pcloud.to(args.device).float())
            pred_class = torch.argmax(output, dim=1).cpu().numpy()

            for i in range(pcloud.size(0)):
                pred = pred_class[i]
                target = label[i].numpy()

                for cls in range(class_num):
                    pred_mask = (pred == cls)
                    target_mask = (target == cls)

                    intersection = (pred_mask & target_mask).sum()
                    union = (pred_mask | target_mask).sum()

                    if union == 0:
                        continue
                    iou = intersection / union
                    iou_list[cls].append(iou)

    class_ious = [np.mean(iou_list[i]) if len(iou_list[i]) > 0 else 0.0 for i in range(class_num)]
    miou = np.mean(class_ious)
    return class_ious, miou


def compute_pcloud_classification_metrics(args, val_dataloader, model):
    acc = 0.0
    val_num = len(val_dataloader.dataset)
    device = args.device
    with torch.no_grad():
        for pcloud, label in tqdm(val_dataloader):
            output = model(pcloud.to(device))
            pred_class = torch.argmax(output, dim=1)
            acc += torch.eq(pred_class, label.to(device)).sum().item()
    acc = acc / val_num
    return acc