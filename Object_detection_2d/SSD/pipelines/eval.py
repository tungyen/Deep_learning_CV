from tqdm import tqdm
import argparse
import numpy as np
import os
import torch
import torch.optim as optim

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from Object_detection_2d.data import build_dataloader
from Object_detection_2d.SSD.model import build_model, PostProcessor
from Object_detection_2d.SSD.utils.config_utils import parse_config
from Object_detection_2d.SSD.utils.ddp_utils import synchronize, gather_preds_ddp, is_main_process
from Object_detection_2d.metrics import compute_object_detection_metrics

def eval_model(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    config_path = args.config
    ckpts_path = args.experiment
    args = parse_config(config_path)
    root = args['root']
    model_name = args['model']
    dataset_type = args['datasets']['name']
    exec_path = os.path.join(root, "runs")
    os.makedirs(exec_path, exist_ok=True)
    os.makedirs(os.path.join(exec_path, ckpts_path), exist_ok=True)
    weight_path = os.path.join(exec_path, ckpts_path, "{}_{}.pt".format(args['model'], args['datasets']['name']))

    if is_main_process():
        print("Start evaluating model {} on {} dataset!".format(args['model'], args['datasets']['name']))
    _, val_dataloader, _ = build_dataloader(args)
    class_dict = val_dataloader.dataset.class_dict
    model = build_model(args).to(local_rank)
    model.load_state_dict(torch.load(weight_path, map_location=f"cuda:{local_rank}"))
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Validation
    model.eval()
    pred_results = {}

    with torch.no_grad():
        for imgs, targets, img_ids in tqdm(val_dataloader, desc=f"Evaluate", disable=not is_main_process()):
            imgs = imgs.to(local_rank)
            with torch.no_grad():
                detections = model(imgs, False)
                detections = [d.to(torch.device("cpu")) for d in detections]
            pred_results.update(
                {int(img_id): d for img_id, d in zip(img_ids, detections)}
            )
    synchronize()
    pred_results = gather_preds_ddp(pred_results)

    if is_main_process():
        metrics = compute_object_detection_metrics(val_dataloader.dataset, pred_results)
        print("Validation mAP of {} on {} ===> {:.4f}".format(args['model'], args['datasets']['name'], metrics['map']))
        for i, ap in enumerate(metrics['ap']):
            if i == 0:
                continue
            print("{} ap: {:.4f}".format(class_dict[i], ap))

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--experiment', type=str, required=True)
    parse.add_argument('--config', type=str, required=True)
    args = parse.parse_args()
    return args
        
if __name__ =='__main__':
    args = parse_args()
    eval_model(args)