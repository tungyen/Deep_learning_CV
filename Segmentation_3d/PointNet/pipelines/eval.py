import torch
from tqdm import tqdm
import os
import argparse

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from Segmentation_3d.data import build_dataloader
from Segmentation_3d.PointNet.model import build_model
from Segmentation_3d.metrics import build_metrics
from Segmentation_3d.utils import is_main_process, parse_config, gather_all_data

def eval_model(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    config_path = args.config_path
    exp = args.exp
    opts = parse_config(config_path)

    root = opts.root
    os.makedirs(os.path.join(root, 'runs'), exist_ok=True)
    os.makedirs(os.path.join(root, 'runs', exp), exist_ok=True)
    weight_path = os.path.join(root, 'runs', exp, "max-iou-val.pth")

    if is_main_process():
        print("Start evaluating model {}!".format(opts.model.name))

    model_name = opts.model.name
    dataset_type = opts.dataset_name
    task = opts.task

    _, val_dataloader, _ = build_dataloader(opts)
    model = build_model(opts.model).to(local_rank)
    model.load_state_dict(torch.load(weight_path, map_location=f"cuda:{local_rank}"))
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    metrics = build_metrics(opts.metrics)
    class_dict = val_dataloader.dataset.get_class_dict()
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for pclouds, *labels in tqdm(val_dataloader, desc="Evaluation", disable=dist.get_rank() != 0):
            # Semantic Segmentation or Classification
            if len(labels) == 1:
                labels = labels[0]
                outputs, _ = model(pclouds.to(local_rank))
                pred_classes = torch.argmax(outputs, dim=1).cpu()
            # Part Segmentation
            elif len(labels) == 2:
                cls_labels, labels = labels
                instance2parts, _, label2class = class_dict
                outputs, _ = model(pclouds.to(local_rank), cls_labels.to(local_rank))
                pred_classes = torch.zeros((outputs.shape[0], outputs.shape[2])).to(local_rank)
                for i in range(outputs.shape[0]):
                    instance = label2class[cls_labels[i].item()]
                    logits = outputs[i, :, :]
                    pred_classes[i, :] = torch.argmax(logits[instance2parts[instance], :], 0) + instance2parts[instance][0]
                all_preds.append(pred_classes.cpu())
                all_labels.append(labels)
            else:
                raise ValueError(f'Too much input data.')
            metrics.update(pred_classes.cpu(), labels)
    all_preds = gather_all_data(all_preds)
    all_labels = gather_all_data(all_labels)
    metrics.gather(local_rank)
    if is_main_process():
        metrics_results = metrics.compute_metrics()
        if task == "cls":
            precision = metrics_results['mean_precision']
            recall = metrics_results['mean_recall']
            print("Validation Precision of {} on {} ===> {:.4f}".format(model_name, dataset_type, precision))
            print("Validation Recall of {} on {} ===> {:.4f}".format(model_name, dataset_type, recall))

        elif task == "semseg":
            ious = metrics_results['ious']
            mious = metrics_results['mious']
            print("Validation mIoU of {} on {} ===> {:.4f}".format(model_name, dataset_type, mious))
            for cls in class_dict:
                print("{} IoU: {:.4f}".format(class_dict[cls], ious[cls]))

        elif task == 'partseg':
            instance_ious, instance_mious, class_mious = compute_pcloud_partseg_metrics(all_preds, all_labels, class_dict)
            print("Validation Instance mIoU of {} on {} ===> {:.4f}".format(model_name, dataset_type, instance_mious))
            print("Validation Class mIoU of {} on {} ===> {:.4f}".format(model_name, dataset_type, class_mious))
            for cls in instance_ious:
                print("{} IoU: {:.4f}".format(cls, instance_ious[cls]))
        else:
            raise ValueError(f'Unknown segmentation task {task}.')  
    dist.destroy_process_group()

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--exp', type=str, required=True)
    parse.add_argument('--config_path', type=str, required=True)
    args = parse.parse_args()
    return args
        
if __name__ =='__main__':
    args = parse_args()
    eval_model(args)