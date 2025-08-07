from tqdm import tqdm
import argparse
import numpy as np
import os
import torch
import torch.optim as optim

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from Object_detection_2d.dataset.utils import get_dataset
from Object_detection_2d.loss import get_loss
from Object_detection_2d.optimizer import get_scheduler
from Object_detection_2d.utils import get_model, setup_args_with_dataset

def train_model(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    ckpts_path = args.experiment
    root = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(root, ckpts_path), exist_ok=True)
    model_name = args.model
    dataset_type = args.dataset
    setup_args_with_dataset(dataset_type, args)

    weight_path = os.path.join(root, ckpts_path, "{}_{}.pth".format(model_name, dataset_type))
    lr = args.lr
    momentum = args.momentum
    epochs = args.epochs
    weight_decay = args.weight_decay

    if dist.get_rank() == 0:
        print("Start training model {} on {} dataset!".format(model_name, dataset_type))
    train_dataloader, val_dataloader, _, class_dict, _, _ = get_dataset(args)
    model = get_model(args).to(local_rank)

    biases = list()
    others = list()
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name.endswith(".bias"):
                biases.append(param)
            else:
                others.append(param)
    optimizer = optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': others, 'lr': lr}],
                          weight_decay=weight_decay, momentum=momentum)
    scheduler = get_scheduler(args, optimizer)
    criterion = get_loss(args, model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    best_metric = 0.0
            
    for epoch in range(epochs):
        # Train
        train_dataloader.sampler.set_epoch(epoch)
        model.train()
        
        with tqdm(train_dataloader, desc=f"Train Epoch {epoch+1}", disable=dist.get_rank() != 0) as pbar:
            for imgs, targets in pbar:
                imgs = imgs.to(local_rank)
                boxes = [t['bboxes'].to(local_rank) for t in targets]
                labels = [t['labels'].to(local_rank) for t in targets]
                pred_boxes, pred_logits = model(imgs)
                loss = criterion(pred_boxes, pred_logits, boxes, labels)

                optimizer.zero_grad()
                loss['loss'].backward()
                optimizer.step()
                if dist.get_rank() == 0:
                    pbar.set_postfix(
                        total_loss=f"{loss['loss'].item():.4f}",
                        cls_loss=f"{loss['cls_loss'].item():.4f}",
                        boxes_loss=f"{loss['boxes_loss'].item():.4f}",
                    )
            scheduler.step()
        # Validation

def parse_args():
    parse = argparse.ArgumentParser()
    # Dataset
    parse.add_argument('--dataset', type=str, default="voc")
    parse.add_argument('--crop_size', type=int, default=[513, 513])
    parse.add_argument('--voc_data_root', type=str, default="Dataset/VOC")
    parse.add_argument('--voc_year', type=str, default="2012")
    parse.add_argument('--voc_download', type=bool, default=False)
    
    # Model
    parse.add_argument('--model', type=str, default="ssd")
    
    # Training
    parse.add_argument('--experiment', type=str, required=True)
    parse.add_argument('--epochs', type=int, default=200)
    parse.add_argument('--scheduler', type=str, default="poly")
    parse.add_argument('--lr', type=float, default=0.01)
    parse.add_argument('--weight_decay', type=float, default=1e-4)
    parse.add_argument('--momentum', type=float, default=0.9)
    parse.add_argument('--step_size', type=int, default=70)
    parse.add_argument('--loss_func', type=str, default="ce_smooth_l1")
    args = parse.parse_args()
    return args
     
if __name__ =='__main__':
    args = parse_args()
    train_model(args)
    
    
    
    