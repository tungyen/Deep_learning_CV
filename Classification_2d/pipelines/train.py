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
from core.utils import is_main_process, init_ddp, save_checkpoint, load_checkpoint

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
    checkpoint_path = os.path.join(root, 'runs', exp, "last-checkpoint.pth")

    if is_main_process():
        config_save_path = os.path.join(root, 'runs', exp, 'config.txt')
        with open(config_save_path, 'w') as f:
            json.dump(opts, f, indent=4)

    if is_main_process():
        print("Start training model {}!".format(opts.model.name))
    
    train_dataloader, val_dataloader, _ = build_dataloader(opts)
    epochs = opts.epochs
    model_name = opts.model.name
    model = build_model(opts.model).to(local_rank)

    opts.scheduler.train_size = len(train_dataloader)
    opts.scheduler.epochs = epochs
    opts.scheduler.world_size = world_size

    optimizer = build_optimizer(opts.optimizer, model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    scheduler = build_scheduler(opts.scheduler, optimizer)
    criterion = build_loss(opts.loss)
    class_dict = val_dataloader.dataset.get_class_dict()
    metrics = build_metrics(class_dict, val_dataloader.dataset, opts.metrics)

    best_metrics = 0.0
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
                imgs = input_dict['img'].to(local_rank)
                labels = input_dict['label'].type(torch.LongTensor).to(local_rank)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
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
        with torch.no_grad():
            for input_dict in tqdm(val_dataloader, desc=f"Evaluate Epoch {epoch+1}", disable=not is_main_process()):
                outputs = model(input_dict['img'].to(local_rank))
                pred_classes = torch.argmax(outputs, dim=1)
                metrics.update(pred_classes.cpu(), input_dict)

        metrics.gather(local_rank)
        if is_main_process():
            metrics_results = metrics.compute_metrics()
            if metrics_results > best_metrics:
                best_metrics = metrics_results
                torch.save(model.module.state_dict(), weight_path)
            save_checkpoint(model, optimizer, scheduler, epoch, best_metric, checkpoint_path)
        metrics.reset()
    dist.destroy_process_group() 

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--exp', type=str, required=True)
    parse.add_argument('--config_path', type=str, required=True)
    parse.add_argument('--resume', default=True)
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    train_model(args)
