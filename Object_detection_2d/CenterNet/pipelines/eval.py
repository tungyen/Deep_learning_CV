from tqdm import tqdm
import argparse
import numpy as np
import os
import torch
import torch.optim as optim

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from core.metrics import build_metrics
from core.utils import is_main_process

from Object_detection_2d.data import build_dataloader
from Object_detection_2d.CenterNet.model import build_model, PostProcessor
from Object_detection_2d.utils import parse_config

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
    model_name = opts.model.name
    dataset_type = opts.datasets.dataset_name
    exec_path = os.path.join(root, "runs")
    weight_path = os.path.join(root, 'runs', exp, "max-ap-val.pt")

    if is_main_process():
        print("Start evaluating model {} on {} dataset!".format(model_name, dataset_type))
    _, val_dataloader, _ = build_dataloader(opts)
    
    class_dict = val_dataloader.dataset.class_dict
    metrics = build_metrics(class_dict, val_dataloader.dataset, opts.metrics)

    model = build_model(opts.model).to(local_rank)
    model.load_state_dict(torch.load(weight_path, map_location=f"cuda:{local_rank}"))
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model.eval()

    for imgs, targets, img_ids in tqdm(val_dataloader, desc=f"Evaluate", disable=not is_main_process()):
        imgs = imgs.to(local_rank)
        with torch.no_grad():
            detections = model(imgs, False)
            detections = [d.to(torch.device("cpu")) for d in detections]
        metrics.update(img_ids, detections)
    metrics.gather(local_rank)

    if is_main_process():
        print("Start computing metrics.")
        metrics_results = metrics.compute_metrics()
        

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--exp', type=str, required=True)
    parse.add_argument('--config_path', type=str, required=True)
    args = parse.parse_args()
    return args

if __name__ =='__main__':
    args = parse_args()
    eval_model(args)
    
    
    
    