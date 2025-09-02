from tqdm import tqdm
import argparse
import numpy as np
import os
import torch
import torch.optim as optim

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from Object_detection_2d.data import build_dataloader
from Object_detection_2d.SSD.loss import build_loss
from Object_detection_2d.optimizer import build_optimizer
from Object_detection_2d.scheduler import build_scheduler
from Object_detection_2d.SSD.model import build_model, PostProcessor
from Object_detection_2d.SSD.utils.config_utils import parse_config
from Object_detection_2d.SSD.utils.ddp_utils import synchronize, gather_preds_ddp, is_main_process
from Object_detection_2d.metrics import compute_object_detection_metrics

def train_model(args):
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
        print("Start training model {} on {} dataset!".format(args['model'], args['datasets']['name']))
    train_dataloader, val_dataloader, _ = build_dataloader(args)
    class_dict = val_dataloader.dataset.class_dict
    model = build_model(args).to(local_rank)

    args['optimizer']['lr'] *= world_size
    args['scheduler']['milestones'] = [m // world_size for m in args['scheduler']['milestones']]
    optimizer = build_optimizer(args, model.parameters())
    scheduler = build_scheduler(args, optimizer)
    criterion = build_loss(args)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    best_metric = 0.0
            
    for epoch in range(args['epochs']):
        # Train
        train_dataloader.sampler.set_epoch(epoch)
        model.train()
        with tqdm(train_dataloader, desc=f"Train Epoch {epoch+1}", disable=not is_main_process()) as pbar:
            for imgs, targets, _ in pbar:
                imgs = imgs.to(local_rank)
                targets = targets.to(local_rank)
                boxes = targets['boxes']
                labels = targets['labels']
                pred_boxes, pred_logits = model(imgs)
                loss = criterion(pred_boxes, pred_logits, boxes, labels)
                optimizer.zero_grad()
                loss['loss'].backward()

                optimizer.step()
                if dist.get_rank() == 0:
                    pbar.set_postfix(
                        total_loss=f"{loss['loss'].item():.4f}",
                        cls_loss=f"{loss['cls_loss'].item():.4f}",
                        boxes_loss=f"{loss['boxes_loss'].item():.4f}"
                    )
                scheduler.step()
        torch.cuda.empty_cache()
        # Validation
        model.eval()
        pred_results = {}

        for imgs, targets, img_ids in tqdm(val_dataloader, desc=f"Evaluate Epoch {epoch+1}", disable=not is_main_process()):
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
            print("Start computing metrics.")
            metrics = compute_object_detection_metrics(val_dataloader.dataset, pred_results)
            print("Validation mAP of {} on {} ===> {:.4f}".format(args['model'], args['datasets']['name'], metrics['map']))
            for i, ap in enumerate(metrics['ap']):
                if i == 0:
                    continue
                print("{} ap: {:.4f}".format(class_dict[i], ap))
            if metrics['map'] > best_metric:
                best_metric = metrics['map']
                torch.save(model.module.state_dict(), weight_path)

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--experiment', type=str, required=True)
    parse.add_argument('--config', type=str, required=True)
    args = parse.parse_args()
    return args

if __name__ =='__main__':
    args = parse_args()
    train_model(args)
    
    
    
    