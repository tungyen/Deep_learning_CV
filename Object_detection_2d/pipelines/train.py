from tqdm import tqdm
import argparse
import numpy as np
import os
import torch

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from core.optimizer import build_optimizer
from core.scheduler import build_scheduler
from core.metrics import build_metrics
from core.utils import is_main_process, init_ddp, save_checkpoint, load_checkpoint

from Object_detection_2d.data import build_dataloader
from Object_detection_2d.loss import build_loss
from Object_detection_2d.model import build_model
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
    checkpoint_path = os.path.join(root, 'runs', exp, "last-checkpoint.pth")

    if is_main_process():
        config_save_path = os.path.join(root, 'runs', exp, 'config.txt')
        with open(config_save_path, 'w') as f:
            json.dump(opts, f, indent=4)

    if is_main_process():
        print("Start training model {} on {} dataset!".format(model_name, dataset_type))
    train_dataloader, val_dataloader, _ = build_dataloader(opts)
    train_size = len(train_dataloader)
    model = build_model(opts.model).to(local_rank)
    epochs = opts.epochs

    optimizer = build_optimizer(opts.optimizer, model)

    opts.scheduler.train_size = train_size
    opts.scheduler.world_size = world_size
    opts.scheduler.epochs = epochs
    scheduler = build_scheduler(opts.scheduler, optimizer)
    criterion = build_loss(opts.loss)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    class_dict = val_dataloader.dataset.get_class_dict()
    metrics = build_metrics(class_dict, val_dataloader.dataset, opts.metrics)

    best_metric = 0.0
    start_epoch = 0

    if args.resume and os.path.exists(checkpoint_path):
        start_epoch, best_metric = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, local_rank
        )
        if is_main_process():
            print(f"Resumed from epoch {start_epoch}, best_metric={best_metric:.4f}")
            
    for epoch in range(start_epoch, epochs):
        # Train
        train_dataloader.sampler.set_epoch(epoch)
        model.train()
        with tqdm(train_dataloader, desc=f"Train Epoch {epoch+1}", disable=not is_main_process()) as pbar:
            for input_dict in pbar:
                input_dict = input_dict.to(local_rank)
                pred = model(input_dict['img'])
                loss = criterion(pred, input_dict)
                optimizer.zero_grad()
                loss['loss'].backward()
                optimizer.step()

                lr_groups = {}
                if len(optimizer.param_groups) == 1:
                    lr_groups['lr'] = optimizer.param_groups[0]['lr']
                else:
                    for i, group in enumerate(optimizer.param_groups):
                        key = f"lr_{group.get('name')}"
                        lr_groups[key] = group['lr']

                postfix = {k: f"{v:.2e}" for k, v in lr_groups.items()}

                for loss_name, loss_value in loss.items():
                    postfix[loss_name] = f"{loss_value.item():.4f}"

                if is_main_process():
                    pbar.set_postfix(postfix)
                scheduler.step()

        # Validation
        model.eval()

        for input_dict in tqdm(val_dataloader, desc=f"Evaluate Epoch {epoch+1}", disable=not is_main_process()):
            imgs = input_dict['img'].to(local_rank)
            with torch.no_grad():
                detections = model(imgs, False)
                detections = [d.to(torch.device("cpu")) for d in detections]
            metrics.update(input_dict, detections)

        metrics.gather(local_rank)
        if is_main_process():
            metrics_results = metrics.compute_metrics()
            if metrics_results > best_metric:
                best_metric = metrics_results
                torch.save(model.module.state_dict(), weight_path)
            save_checkpoint(model, optimizer, scheduler, epoch, best_metric, checkpoint_path)
        metrics.reset()

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--exp', type=str, required=True)
    parse.add_argument('--config_path', type=str, required=True)
    parse.add_argument('--resume', default=True)
    args = parse.parse_args()
    return args

if __name__ =='__main__':
    args = parse_args()
    train_model(args)
    
    
    
    