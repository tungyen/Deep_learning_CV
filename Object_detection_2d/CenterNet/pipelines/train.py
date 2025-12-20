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
from core.utils import is_main_process

from Object_detection_2d.data import build_dataloader
from Object_detection_2d.CenterNet.loss import build_loss
from Object_detection_2d.CenterNet.model import build_model, PostProcessor
from Object_detection_2d.utils import parse_config

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
        torch.cuda.empty_cache()
        # Validation
        # model.eval()
        # pred_results = {}

        # for imgs, targets, img_ids in tqdm(val_dataloader, desc=f"Evaluate Epoch {epoch+1}", disable=not is_main_process()):
        #     imgs = imgs.to(local_rank)
        #     with torch.no_grad():
        #         detections = model(imgs, False)
        #         detections = [d.to(torch.device("cpu")) for d in detections]
        #     pred_results.update(
        #         {int(img_id): d for img_id, d in zip(img_ids, detections)}
        #     )

        # synchronize()
        # pred_results = gather_preds_ddp(pred_results)

        if is_main_process():
            # print("Start computing metrics.")
            # metrics = compute_object_detection_metrics(val_dataloader.dataset, pred_results)
            # print("Validation mAP of {} on {} ===> {:.4f}".format(model_name, dataset_type, metrics['map']))
            # for i, ap in enumerate(metrics['ap']):
            #     if i == 0:
            #         continue
            #     print("{} ap: {:.4f}".format(class_dict[i], ap))
            # if metrics['map'] > best_metric:
            #     best_metric = metrics['map']
            #     torch.save(model.module.state_dict(), weight_path)
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
    
    
    
    