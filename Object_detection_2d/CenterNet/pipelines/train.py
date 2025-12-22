from tqdm import tqdm
import argparse
import numpy as np
import os
import torch
import torch.optim as optim

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from core.optimizer import build_optimizer
from core.scheduler import build_scheduler
from core.metrics import build_metrics
from core.utils import is_main_process, init_ddp

from Object_detection_2d.data import build_dataloader
from Object_detection_2d.CenterNet.loss import build_loss
from Object_detection_2d.CenterNet.model import build_model, PostProcessor
from Object_detection_2d.utils import parse_config

def train_model(args):
    local_rank, rank, world_size = init_ddp()
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
        print("Start training model {} on {} dataset!".format(model_name, dataset_type))
    train_dataloader, val_dataloader, _ = build_dataloader(opts)
    class_dict = val_dataloader.dataset.class_dict
    model = build_model(opts.model).to(local_rank)

    epochs = opts.epochs
    opts.optimizer.lr *= world_size
    train_size = len(train_dataloader)

    if opts.scheduler.name == "CosineAnnealingWarmup":
        warmup_epochs = opts.scheduler.pop('warmup_epochs', None)
        opts.scheduler.first_cycle_steps = train_size * (epochs - warmup_epochs)
        opts.scheduler.warmup_steps = train_size * warmup_epochs
    optimizer = build_optimizer(opts.optimizer, model.parameters())
    scheduler = build_scheduler(opts.scheduler, optimizer)
    criterion = build_loss(opts.loss)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    class_dict = val_dataloader.dataset.get_class_dict()
    metrics = build_metrics(class_dict, val_dataloader.dataset, opts.metrics)
    best_metric = 0.0
            
    for epoch in range(epochs):
        # Train
        train_dataloader.sampler.set_epoch(epoch)
        model.train()
        with tqdm(train_dataloader, desc=f"Train Epoch {epoch+1}", disable=not is_main_process()) as pbar:
            for imgs, targets, _ in pbar:
                imgs = imgs.to(local_rank)
                targets = targets.to(local_rank)
                pred_dict = model(imgs)
                loss = criterion(pred_dict, targets)
                optimizer.zero_grad()
                loss['loss'].backward()

                optimizer.step()
                if dist.get_rank() == 0:
                    pbar.set_postfix(
                        total_loss=f"{loss['loss'].item():.4f}",
                        hs_loss=f"{loss['hm_loss'].item():.4f}",
                        wh_loss=f"{loss['wh_loss'].item():.4f}",
                        offset_loss=f"{loss['offset_loss'].item():.4f}"
                    )
                scheduler.step()

        # Validation
        model.eval()

        for imgs, targets, img_ids in tqdm(val_dataloader, desc=f"Evaluate Epoch {epoch+1}", disable=not is_main_process()):
            imgs = imgs.to(local_rank)
            with torch.no_grad():
                detections = model(imgs, False)
                detections = [d.to(torch.device("cpu")) for d in detections]
            metrics.update(img_ids, detections)
        metrics.gather(local_rank)

        if is_main_process():
            metrics_results = metrics.compute_metrics()
            if metrics_results > best_metric:
                best_metric = metrics_results
                torch.save(model.module.state_dict(), weight_path)

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--exp', type=str, required=True)
    parse.add_argument('--config_path', type=str, required=True)
    args = parse.parse_args()
    return args

if __name__ =='__main__':
    args = parse_args()
    train_model(args)
    
    
    
    