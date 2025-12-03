import os
import torch
import numpy as np
from tqdm import tqdm
import argparse

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from Segmentation_2d.data import build_dataloader
from Segmentation_2d.DeepLabV3.model import build_model
from Segmentation_2d.optimizer import build_optimizer
from Segmentation_2d.scheduler import build_scheduler
from Segmentation_2d.metrics import build_metrics
from Segmentation_2d.loss import build_loss
from Segmentation_2d.utils import all_reduce_confusion_matrix, is_main_process, parse_config

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
    
    train_dataloader, val_dataloader, _ = build_dataloader(opts)
    epochs = opts.epochs
    model_name = opts.model.name
    model = build_model(opts.model).to(local_rank)
    optimizer = build_optimizer(opts.optimizer, model.parameters())
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    scheduler = build_scheduler(opts.scheduler, optimizer)
    criterion = build_loss(opts.loss)
    metrics = build_metrics(opts.metrics)

    best_metric = 0.0
    world_size = dist.get_world_size()
    class_dict = val_dataloader.dataset.get_class_dict()

    for epoch in range(epochs):
        train_dataloader.sampler.set_epoch(epoch)
        model.train()
        
        # Train
        with tqdm(train_dataloader, desc=f"Train Epoch {epoch+1}", disable=not is_main_process()) as pbar:
            for imgs, labels in pbar:
                imgs = imgs.to(local_rank)
                labels = labels.type(torch.LongTensor).to(local_rank)
                outputs = model(imgs)

                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss['loss'].backward()
                optimizer.step()
                if dist.get_rank() == 0:
                    pbar.set_postfix(
                        total_loss=f"{loss['loss'].item():.4f}",
                        ce_loss=f"{loss['ce_loss'].item():.4f}",
                        lovasz_softmax_loss=f"{loss['lovasz_softmax_loss'].item():.4f}" if 'lovasz_softmax_loss' in loss else "0.0000",
                        boundary_loss=f"{loss['boundary_loss'].item():.4f}" if 'boundary_loss' in loss else "0.0000"
                    )
                scheduler.step()
        # Validation
        model.eval()
        for imgs, labels in tqdm(val_dataloader, desc=f"Evaluate Epoch {epoch+1}", disable=dist.get_rank() != 0):
            with torch.no_grad():
                outputs = model(imgs.to(local_rank))
                pred_class = torch.argmax(outputs, dim=1)
                metrics.update(pred_class.cpu(), labels)

        all_reduce_confusion_matrix(metrics, local_rank)
        if is_main_process:
            metrics_results = metrics.compute_metrics()
            print("Validation mIoU  ===>{:.4f}".format(metrics['mious'].item()))
            for i, iou in enumerate(metrics['ious']):
                print("{} IoU: {:.4f}".format(class_dict[i], iou))
                
            if metrics['mious'] > best_metric:
                best_metric = metrics['mious']
                torch.save(model.module.state_dict(), weight_path)
        metrics.reset()
    dist.destroy_process_group()
    
def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--exp', type=str, required=True)
    parse.add_argument('--config_path', type=str, required=True)
    args = parse.parse_args()
    return args

if __name__ =='__main__':
    args = parse_args()
    train_model(args)