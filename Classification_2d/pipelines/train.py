import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from core.optimizer import build_optimizer
from core.scheduler import build_scheduler
from core.metrics import build_metrics
from core.utils import is_main_process, init_ddp

from Classification_2d.data import build_dataloader
from Classification_2d.model import build_model
from Classification_2d.loss import build_loss
from Classification_2d.utils import parse_config

def train_model(args):
    local_rank, rank, world_size = init_ddp()
    config_path = args.config_path
    exp = args.exp
    opts = parse_config(config_path)

    root = opts.root
    os.makedirs(os.path.join(root, 'runs'), exist_ok=True)
    os.makedirs(os.path.join(root, 'runs', exp), exist_ok=True)
    weight_path = os.path.join(root, 'runs', exp, "max-f1-val.pth")

    if is_main_process():
        print("Start training model {}!".format(opts.model.name))
    
    train_dataloader, val_dataloader, _ = build_dataloader(opts)
    epochs = opts.epochs
    model_name = opts.model.name
    model = build_model(opts.model).to(local_rank)
    optimizer = build_optimizer(opts.optimizer, model.parameters())
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    scheduler = build_scheduler(opts.scheduler, optimizer)
    criterion = build_loss(opts.loss)
    class_dict = val_dataloader.dataset.get_class_dict()
    metrics = build_metrics(class_dict, val_dataloader.dataset, opts.metrics)

    best_metrics = 0.0
    for epoch in range(epochs):
        # Train
        train_dataloader.sampler.set_epoch(epoch)
        model.train()

        with tqdm(train_dataloader, desc=f"Train Epoch {epoch+1}", disable=not is_main_process()) as pbar:
            for imgs, labels in pbar:
                outputs = model(imgs.to(local_rank))
                loss = criterion(outputs, labels.to(local_rank))
                optimizer.zero_grad()
                loss['loss'].backward()
                optimizer.step()
                scheduler.step()
                pbar.set_postfix(total_loss=f"{loss['loss'].item():.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            for imgs, labels in tqdm(val_dataloader, desc=f"Evaluate Epoch {epoch+1}", disable=not is_main_process()):
                outputs = model(imgs.to(local_rank))
                pred_classes = torch.argmax(outputs, dim=1)
                metrics.update(pred_classes.cpu(), labels)

        metrics.gather(local_rank)
        if is_main_process():
            metrics_results = metrics.compute_metrics()
            if metrics_results > best_metrics:
                best_metrics = metrics_results
                torch.save(model.module.state_dict(), weight_path)
        metrics.reset()
    dist.destroy_process_group() 

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--exp', type=str, required=True)
    parse.add_argument('--config_path', type=str, required=True)
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    train_model(args)
