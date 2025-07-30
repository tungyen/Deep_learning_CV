import os
import torch
import numpy as np
from tqdm import tqdm
import argparse

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from Segmentation_2d.optimizer import get_scheduler
from Segmentation_2d.loss import get_loss
from Segmentation_2d.dataset.utils import get_dataset
from Segmentation_2d.utils import get_model, setup_args_with_dataset, all_reduce_confusion_matrix
from Segmentation_2d.metrics import ConfusionMatrix

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
    args = setup_args_with_dataset(dataset_type, args)
    
    if dataset_type == 'cityscapes':
        weight_path = os.path.join(root, ckpts_path, "{}_{}.pth".format(model_name, dataset_type))
    elif dataset_type == 'voc':
        weight_path = os.path.join(root, ckpts_path, "{}_{}_{}.pth".format(model_name, dataset_type, args.voc_year))
    else:
        raise ValueError(f'Unknown dataset {dataset_type}.')

    lr = args.lr
    epochs = args.epochs
    weight_decay = args.weight_decay
    momentum = args.momentum
    
    model = get_model(args).to(local_rank)
    # model.load_state_dict(torch.load(weight_path, map_location=f"cuda:{local_rank}"))
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    train_dataloader, val_dataloader, _, class_dict, _, _ = get_dataset(args)
    optimizer = torch.optim.SGD(params=[
        {'params': model.module.backbone.parameters(), 'lr': 0.1 * lr},
        {'params': model.module.classifier.parameters(), 'lr': lr},
    ], lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = get_scheduler(args, optimizer)
    criterion = get_loss(args)
    confusion_matrix = ConfusionMatrix(class_num=args.class_num, ignore_index=args.ignore_idx)
    
    if dist.get_rank() == 0:
        print("Start training model {} on {} dataset!".format(model_name, dataset_type))

    best_metric = 0.0
    world_size = dist.get_world_size()

    for epoch in range(epochs):
        train_dataloader.sampler.set_epoch(epoch)
        model.train()
        
        # Train
        with tqdm(train_dataloader, desc=f"Train Epoch {epoch+1}", disable=dist.get_rank() != 0) as pbar:
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
                confusion_matrix.update(pred_class.cpu(), labels)

        all_reduce_confusion_matrix(confusion_matrix, local_rank)
        if dist.get_rank() == 0:
            metrics = confusion_matrix.compute_metrics()
            print("Validation mIoU of {} on {} ===>{:.4f}".format(model_name, dataset_type, metrics['mious'].item()))
            for i, iou in enumerate(metrics['ious']):
                print("{} IoU: {:.4f}".format(class_dict[i], iou))
                
            if metrics['mious'] > best_metric:
                best_metric = metrics['mious']
                torch.save(model.module.state_dict(), weight_path)
        confusion_matrix.reset()
    dist.destroy_process_group()
    
def parse_args():
    parse = argparse.ArgumentParser()
    # Dataset
    parse.add_argument('--dataset', type=str, default="cityscapes")
    parse.add_argument('--crop_size', type=list, default=[512, 512])
    parse.add_argument('--voc_data_root', type=str, default="Dataset/VOC")
    parse.add_argument('--voc_year', type=str, default="2012_aug")
    parse.add_argument('--voc_download', type=bool, default=False)
    parse.add_argument('--voc_crop_val', type=bool, default=True)
    parse.add_argument('--cityscapes_crop_val', type=bool, default=True)
    
    # Model
    parse.add_argument('--model', type=str, default="deeplabv3")
    parse.add_argument('--backbone', type=str, default="resnet101")
    parse.add_argument('--bn_momentum', type=float, default=0.1)
    
    # Training
    parse.add_argument('--experiment', type=str, required=True)
    parse.add_argument('--epochs', type=int, default=100)
    parse.add_argument('--scheduler', type=str, default="poly")
    parse.add_argument('--lr', type=float, default=0.01)
    parse.add_argument('--weight_decay', type=float, default=1e-4)
    parse.add_argument('--momentum', type=float, default=0.9)
    parse.add_argument('--step_size', type=int, default=70)
    parse.add_argument('--max_iters', type=int, default=9e4)
    parse.add_argument('--loss_func', type=str, default="ce")
    parse.add_argument('--lovasz_weight', type=float, default=1.5)
    parse.add_argument('--boundary_weight', type=float, default=0.5)
    args = parse.parse_args()
    return args

if __name__ =='__main__':
    args = parse_args()
    train_model(args)