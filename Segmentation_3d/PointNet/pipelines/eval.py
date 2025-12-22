import torch
from tqdm import tqdm
import os
import argparse

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from core.metrics import build_metrics
from core.utils import is_main_process, init_ddp

from Segmentation_3d.data import build_dataloader
from Segmentation_3d.PointNet.model import build_model
from Segmentation_3d.utils import parse_config

def eval_model(args):
    local_rank, rank, world_size = init_ddp()
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

    _, val_dataloader, _ = build_dataloader(opts)
    model = build_model(opts.model).to(local_rank)
    model.load_state_dict(torch.load(weight_path, map_location=f"cuda:{local_rank}"))
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    class_dict = val_dataloader.dataset.get_class_dict()
    metrics = build_metrics(class_dict, val_dataloader.dataset, opts.metrics)
    model.eval()

    with torch.no_grad():
        for pclouds, labels in tqdm(val_dataloader, desc="Evaluation", disable=not is_main_process()):
            if not isinstance(labels, list):
                    outputs, _ = model(pclouds.to(local_rank))
                    pred_classes = torch.argmax(outputs, dim=1).cpu()
            else:
                cls_labels = labels[0]
                labels = labels[1]
                outputs, _ = model(pclouds.to(local_rank), cls_labels.to(local_rank))
                pred_classes = model.module.post_process(outputs, cls_labels, class_dict)
            metrics.update(pred_classes, labels)

    metrics.gather(local_rank)
    if is_main_process():
        results = metrics.compute_metrics()
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