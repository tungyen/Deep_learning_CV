import torch
import torch.optim as optim
from tqdm import tqdm
import argparse
import os
import numpy as np

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from Segmentation_3d.dataset.utils import get_dataset
from Segmentation_3d.utils import get_model, setup_args_with_dataset
from Segmentation_3d.optimizer import get_scheduler
from Segmentation_3d.loss import get_loss
from Segmentation_3d.metrics import compute_pcloud_semseg_metrics, compute_pcloud_cls_metrics, compute_pcloud_partseg_metrics

def train_model(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    ckpts_path = args.experiment
    root = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(root, ckpts_path), exist_ok=True)
    model_name = args.model
    task = model_name[-3:]
    dataset_type = args.dataset
    setup_args_with_dataset(dataset_type, args)
    
    task = args.task
    weight_path = os.path.join(root, ckpts_path, "{}_{}_{}.pth".format(model_name, dataset_type, task))
    lr = args.lr
    beta1= args.beta1
    beta2 = args.beta2
    eps = args.eps
    epochs = args.epochs
    weight_decay = args.weight_decay
    
    if dist.get_rank() == 0:
        print("Start training model {} on {} dataset!".format(model_name, dataset_type))
    train_dataloader, val_dataloader, _, class_dict = get_dataset(args)
    model = get_model(args).to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2), eps=eps)
    scheduler = get_scheduler(args, opt)
    criterion = get_loss(args)
    best_metric = 0.0
    if args.task[-3:] == "seg":
        confusion_matrix = ConfusionMatrix(class_num=args.seg_class_num)
    else:
        confusion_matrix = ConfusionMatrix(class_num=args.cls_class_num)
            
    for epoch in range(epochs):
        train_dataloader.sampler.set_epoch(epoch)
        model.train()
        with tqdm(train_dataloader, desc=f"Train Epoch {epoch+1}", disable=dist.get_rank() != 0) as pbar:
            for pclouds, *labels in pbar:
                pclouds = pclouds.to(local_rank).float()
                if len(labels) == 1:
                    labels = labels[0].to(local_rank)
                    outputs, trans_feats = model(pclouds)
                elif len(labels) == 2:
                    cls_labels, labels = labels
                    cls_labels = cls_labels.to(local_rank)
                    labels = labels.to(local_rank)
                    outputs, trans_feats = model(pclouds, cls_labels)
                else:
                    raise ValueError(f'Too much input data.')

                loss = criterion(outputs, labels, trans_feats)
                opt.zero_grad()
                loss['loss'].backward()
                opt.step()
                if dist.get_rank() == 0:
                    if args.loss_func == "ce":
                        pbar.set_postfix(
                            total_loss=f"{loss['loss'].item():.4f}",
                            ce_loss=f"{loss['ce_loss'].item():.4f}",
                            lovasz_softmax_loss=f"{loss['lovasz_softmax_loss'].item():.4f}" if 'lovasz_softmax_loss' in loss else "0.0000",
                            transform_loss=f"{loss['transform_loss'].item():.4f}" if 'transform_loss' in loss else "0.0000"
                        )
                    elif args.loss_func == "focal":
                        pbar.set_postfix(
                            total_loss=f"{loss['loss'].item():.4f}",
                            focal_loss=f"{loss['focal_loss'].item():.4f}",
                            lovasz_softmax_loss=f"{loss['lovasz_softmax_loss'].item():.4f}" if 'lovasz_softmax_loss' in loss else "0.0000",
                            transform_loss=f"{loss['transform_loss'].item():.4f}" if 'transform_loss' in loss else "0.0000"
                        )
            scheduler.step()    
        
        # Validation
        with torch.no_grad():
            for pclouds, *labels in tqdm(val_dataloader, desc="Evaluation"):
                # Semantic Segmentation or Classification
                if len(labels) == 1:
                    labels = labels[0]
                    outputs, _ = model(pclouds.to(local_rank))
                    pred_classes = torch.argmax(outputs, dim=1).cpu()
                # Part Segmentation
                elif len(labels) == 2:
                    cls_labels, labels = labels
                    instance2parts, _, label2class = class_dict
                    outputs, _ = model(pclouds.to(local_rank), cls_labels.to(local_rank))
                    outputs = outputs.cpu()
                    pred_classes = torch.zeros((outputs.shape[0], outputs.shape[2])).astype(torch.int64)
                    for i in range(outputs.shape[0]):
                        instance = label2class[cls_labels[i].item()]
                        logits = outputs[i, :, :]
                        pred_classes[i, :] = torch.argmax(logits[instance2parts[instance], :], 0) + instance2parts[instance][0]
                else:
                    raise ValueError(f'Too much input data.')
                confusion_matrix.update(pred_classes.cpu(), labels)
        if dist.get_rank() == 0:
            metrics = confusion_matrix.compute_metrics()
            if task == "cls":
                precision = metrics['mean_precision']
                recall = metrics['mean_recall']
                print("Validation Precision of {} on {} ===> {:.4f}".format(model_name, dataset_type, precision))
                print("Validation Recall of {} on {} ===> {:.4f}".format(model_name, dataset_type, recall))

                if precision > best_metric:
                    best_metric = precision
                    torch.save(model.state_dict(), weight_path)
            elif task == "semseg":
                ious = metrics['ious']
                mious = metrics['mious']
                print("Validation mIoU of {} on {} ===> {:.4f}".format(model_name, dataset_type, mious))
                for cls in class_dict:
                    print("{} IoU: {:.4f}".format(class_dict[cls], ious[cls]))
                if mious > best_metric:
                    best_metric = mious
                    torch.save(model.state_dict(), weight_path)
            elif task == 'partseg':
                pass
            else:
                raise ValueError(f'Unknown segmentation task {task}.')
            confusion_matrix.reset() 

def parse_args():
    parse = argparse.ArgumentParser()
    # Dataset
    parse.add_argument('--dataset', type=str, default="s3dis")
    
    # S3DIS
    parse.add_argument('--test_area', type=int, default=5)
    parse.add_argument('--max_dropout', type=float, default=0.95)
    parse.add_argument('--block_type', type=str, default='static')
    parse.add_argument('--block_size', type=float, default=1.0)
    
    # ShapeNet
    parse.add_argument('--normal_channel', type=bool, default=True)
    parse.add_argument('--class_choice', type=list, default=None)
    
    # Model
    parse.add_argument('--model', type=str, default="pointnet")
    
    # training
    parse.add_argument('--experiment', type=str, required=True)
    parse.add_argument('--epochs', type=int, default=200)
    parse.add_argument('--lr', type=float, default=1e-3)
    parse.add_argument('--beta1', type=float, default=0.9)
    parse.add_argument('--beta2', type=float, default=0.999)
    parse.add_argument('--eps', type=float, default=1e-8)
    parse.add_argument('--weight_decay', type=float, default=1e-4)
    parse.add_argument('--scheduler', type=str, default="exp")
    parse.add_argument('--loss_func', type=str, default="focal")
    parse.add_argument('--lovasz_weight', type=float, default=1.5)
    args = parse.parse_args()
    return args
     
if __name__ =='__main__':
    args = parse_args()
    train_model(args)
    
    
    
    