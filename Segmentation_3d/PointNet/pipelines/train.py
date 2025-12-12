import os
import torch
import numpy as np
from tqdm import tqdm
import argparse

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from Segmentation_3d.data import build_dataloader
from Segmentation_3d.PointNet.model import build_model
from Segmentation_3d.optimizer import build_optimizer
from Segmentation_3d.scheduler import build_scheduler
from Segmentation_3d.metrics import build_metrics
from Segmentation_3d.loss import build_loss
from Segmentation_3d.utils import is_main_process, parse_config, gather_all_data

def train_model(args):
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
        print("Start training model {}!".format(opts.model.name))

    model_name = opts.model.name
    dataset_type = opts.dataset_name
    epochs = opts.epochs
    task = opts.task

    train_dataloader, val_dataloader, _ = build_dataloader(opts)
    model = build_model(opts.model).to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = build_optimizer(opts.optimizer, model.parameters())
    scheduler = build_scheduler(opts.scheduler, optimizer)
    criterion = build_loss(opts.loss)
    metrics = build_metrics(opts.metrics)

    best_metric = 0.0
    class_dict = val_dataloader.dataset.get_class_dict()
            
    for epoch in range(epochs):
        train_dataloader.sampler.set_epoch(epoch)
        model.train()

        # Train
        with tqdm(train_dataloader, desc=f"Train Epoch {epoch+1}", disable=not is_main_process()) as pbar:
            for pclouds, labels in pbar:
                if not isinstance(labels, list):
                    labels = labels.to(local_rank)
                    outputs, trans_feats = model(pclouds.to(local_rank))
                else:
                    cls_labels = labels[0]
                    labels = labels[1]
                    labels = labels.to(local_rank)
                    outputs, trans_feats = model(pclouds.to(local_rank), cls_labels.to(local_rank))
                loss = criterion(outputs, labels, trans_feats)
                optimizer.zero_grad()
                loss['loss'].backward()
                optimizer.step()
                if is_main_process():
                    postfix_dict = {k: f"{v.item():.4f}" for k, v in loss.items()}
                    pbar.set_postfix(postfix_dict)
            scheduler.step()    

        # Validation
        with torch.no_grad():
            for pclouds, labels in tqdm(val_dataloader, desc="Evaluation"):
                if not isinstance(labels, list):
                    outputs, _ = model(pclouds.to(local_rank))
                    pred_classes = torch.argmax(outputs, dim=1).cpu()
                else:
                    cls_labels = labels[0]
                    labels = labels[1]
                    outputs, _ = model(pclouds.to(local_rank), cls_labels.to(local_rank))
                    pred_classes = model.module.post_process(outputs, cls_labels, class_dict)

                metrics.update(pred_classes.cpu(), labels)

        metrics.gather(local_rank)
        if is_main_process():
            metrics_results = metrics.compute_metrics(class_dict)
            if metrics_results > best_metric:
                best_metric = metrics_results
                torch.save(model.module.state_dict(), weight_path)
        metrics.reset() 

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--exp', type=str, required=True)
    parse.add_argument('--config_path', type=str, required=True)
    args = parse.parse_args()
    return args
     
if __name__ =='__main__':
    args = parse_args()
    train_model(args)
    
    
    
    