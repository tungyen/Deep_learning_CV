import os
import torch
import numpy as np
from tqdm import tqdm
import argparse

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from core.optimizer import build_optimizer
from core.scheduler import build_scheduler
from core.metrics import build_metrics
from core.utils import is_main_process, init_ddp, , save_checkpoint, load_checkpoint

from Segmentation_3d.data import build_dataloader
from Segmentation_3d.PointNet.model import build_model
from Segmentation_3d.loss import build_loss
from Segmentation_3d.utils import parse_config

def train_model(args):
    local_rank, rank, world_size = init_ddp()
    config_path = args.config_path
    exp = args.exp
    opts = parse_config(config_path)

    root = opts.root
    os.makedirs(os.path.join(root, 'runs'), exist_ok=True)
    os.makedirs(os.path.join(root, 'runs', exp), exist_ok=True)
    weight_path = os.path.join(root, 'runs', exp, "max-iou-val.pth")
    checkpoint_path = os.path.join(root, 'runs', exp, "last-checkpoint.pth")

    if is_main_process():
        config_save_path = os.path.join(root, 'runs', exp, 'config.txt')
        with open(config_save_path, 'w') as f:
            json.dump(opts, f, indent=4)

    if is_main_process():
        print("Start training model {}!".format(opts.model.name))

    model_name = opts.model.name
    dataset_type = opts.dataset_name
    epochs = opts.epochs

    train_dataloader, val_dataloader, _ = build_dataloader(opts)
    model = build_model(opts.model).to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = build_optimizer(opts.optimizer, model)
    scheduler = build_scheduler(opts.scheduler, optimizer)
    criterion = build_loss(opts.loss)
    class_dict = val_dataloader.dataset.get_class_dict()
    metrics = build_metrics(class_dict, None, opts.metrics)

    best_metric = 0.0
    start_epoch = 0

    if args.resume and os.path.exists(checkpoint_path):
        start_epoch, best_metric = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, local_rank
        )
        if is_main_process():
            print(f"Resumed from epoch {start_epoch}, best_metric={best_metric:.4f}")
            
    for epoch in range(start_epoch, epochs):
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
    
    
    
    