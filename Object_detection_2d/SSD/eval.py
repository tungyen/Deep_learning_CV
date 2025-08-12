from tqdm import tqdm
import argparse
import numpy as np
import os
import torch

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from Object_detection_2d.metrics import compute_object_detection_mAP
from Object_detection_2d.dataset.utils import get_dataset
from Object_detection_2d.utils import (
    get_model,
    setup_args_with_dataset,
    decode_boxes,
    gather_list_ddp
)

def eval_model(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    root = os.path.dirname(os.path.abspath(__file__))
    model_name = args.model
    dataset_type = args.dataset
    args = setup_args_with_dataset(dataset_type, args)
    task = args.task
    ckpts_path = args.experiment
    weight_path = os.path.join(root, ckpts_path, "{}_{}_{}.pth".format(model_name, dataset_type, task))
    
    if dist.get_rank() == 0:
        print("Start evaluation model {} on {} dataset!".format(model_name, dataset_type))
    _, val_dataloader, _, class_dict, _, _ = get_dataset(args)
    model = get_model(args).to(local_rank)
    model.load_state_dict(torch.load(weight_path, map_location=f"cuda:{local_rank}"))
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model.eval()
    
    # Validation
    pred_boxes = list()
    pred_labels = list()
    pred_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()
    
    with torch.no_grad():
        for imgs, targets in tqdm(val_dataloader, desc=f"Evaluate Epoch {epoch+1}", disable=dist.get_rank() != 0):
            imgs = imgs.to(local_rank)
            pred_boxes, pred_scores = model(imgs)
            pred_boxes_batch, pred_labels_batch, pred_scores_batch = decode_boxes(
                args, pred_boxes, pred_scores,
                model.module.prior_boxes_center
            )

            boxes = [t['bboxes'].to(local_rank) for t in targets]
            labels = [t['labels'].to(local_rank) for t in targets]
            difficulties = [t['difficulties'].to(local_rank) for t in targets]

            pred_boxes.append(pred_boxes_batch)
            pred_labels.append(pred_labels_batch)
            pred_scores.append(pred_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)
            
        pred_boxes = gather_list_ddp(pred_boxes)
        pred_labels = gather_list_ddp(pred_labels)
        pred_scores = gather_list_ddp(pred_scores)
        true_boxes = gather_list_ddp(true_boxes)
        true_labels = gather_list_ddp(true_labels)
        true_difficulties = gather_list_ddp(true_difficulties)

        if dist.get_rank() == 0:
            APs, mAP = compute_object_detection_mAP(
                args, pred_boxes, pred_labels, pred_scores,
                true_boxes, true_labels, true_difficulties
            )
            print("Validation mAP of {} on {} ===>{:.4f}".format(model_name, dataset_type, mAP.item()))
            for i in range(args.class_num):
                print("{} AP: {:.4f}".format(class_dict[i+1], APs[i].item()))

def parse_args():
    parse = argparse.ArgumentParser()
    # Dataset
    parse.add_argument('--dataset', type=str, default="voc")
    parse.add_argument('--crop_size', type=int, default=[300, 300])
    parse.add_argument('--voc_data_root', type=str, default="Dataset/VOC")
    parse.add_argument('--voc_year', type=str, default="2012")
    parse.add_argument('--voc_download', type=bool, default=False)
    
    # Model
    parse.add_argument('--model', type=str, default="ssd")
    
    # Eval
    parse.add_argument('--experiment', type=str, required=True)
    args = parse.parse_args()
    return args
        
if __name__ =='__main__':
    args = parse_args()
    eval_model(args)