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
    gather_tensors
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
    ckpts_path = args.experiment
    weight_path = os.path.join(root, ckpts_path, "{}_{}.pth".format(model_name, dataset_type))
    
    if dist.get_rank() == 0:
        print("Start evaluation model {} on {} dataset!".format(model_name, dataset_type))
    _, val_dataloader, _, class_dict, _, _ = get_dataset(args)
    model = get_model(args).to(local_rank)
    model.load_state_dict(torch.load(weight_path, map_location=f"cuda:{local_rank}"))
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model.eval()
    
    # Validation
    pred_boxes_all = list()
    pred_labels_all = list()
    pred_scores_all = list()
    pred_counts_all = list()
    true_boxes_all = list()
    true_labels_all = list()
    true_difficulties_all = list()
    true_counts_all = list()
    with torch.no_grad():
        for imgs, targets in tqdm(val_dataloader, desc="Evaluate", disable=dist.get_rank() != 0):
            imgs = imgs.to(local_rank)
            pred_boxes, pred_scores = model(imgs)
            pred_boxes_all_batch, pred_labels_all_batch, pred_scores_all_batch, pred_counts_all_batch = decode_boxes(
                args, pred_boxes, pred_scores,
                model.module.prior_boxes_center
            )

            gt_boxes_all_batch = torch.cat([t['bboxes'].to(local_rank) for t in targets], dim=0)
            gt_labels_all_batch = torch.cat([t['labels'].to(local_rank) for t in targets], dim=0)
            gt_difficulties_all_batch = torch.cat([t['difficulties'].to(local_rank) for t in targets], dim=0)
            gt_counts_all_batch = torch.tensor([len(t['bboxes']) for t in targets], dtype=torch.int32, device=local_rank)

            pred_boxes_all.append(pred_boxes_all_batch)
            pred_labels_all.append(pred_labels_all_batch)
            pred_scores_all.append(pred_scores_all_batch)
            pred_counts_all.append(pred_counts_all_batch)
            true_boxes_all.append(gt_boxes_all_batch)
            true_labels_all.append(gt_labels_all_batch)
            true_difficulties_all.append(gt_difficulties_all_batch)
            true_counts_all.append(gt_counts_all_batch)

        pred_boxes_all = torch.cat(pred_boxes_all, dim=0)
        pred_labels_all = torch.cat(pred_labels_all, dim=0)
        pred_scores_all = torch.cat(pred_scores_all, dim=0)
        pred_counts_all = torch.cat(pred_counts_all, dim=0)
        true_boxes_all = torch.cat(true_boxes_all, dim=0)
        true_labels_all = torch.cat(true_labels_all, dim=0)
        true_difficulties_all = torch.cat(true_difficulties_all, dim=0)
        true_counts_all = torch.cat(true_counts_all, dim=0)

            
        pred_boxes_all = ggather_tensors(pred_boxes_all).cpu()
        pred_labels_all = gather_tensors(pred_labels_all).cpu()
        pred_scores_all = gather_tensors(pred_scores_all).cpu()
        pred_counts_all = gather_tensors(pred_counts_all).cpu()
        true_boxes_all = gather_tensors(true_boxes_all).cpu()
        true_labels_all = gather_tensors(true_labels_all).cpu()
        true_difficulties_all = gather_tensors(true_difficulties_all).cpu()
        true_counts_all = gather_tensors(true_counts_all).cpu()
        print("Merge Completed!")

        if dist.get_rank() == 0:
            APs, mAP = compute_object_detection_mAP(
                args, pred_boxes_all, pred_labels_all, pred_scores_all, pred_counts_all,
                true_boxes_all, true_labels_all, true_difficulties_all, true_counts_all
            )
            print("Validation mAP of {} on {} ===>{:.4f}".format(model_name, dataset_type, mAP.item()))
            for i in range(args.class_num-1):
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
    parse.add_argument('--min_scores', type=float, default=0.01)
    parse.add_argument('--max_overlap', type=float, default=0.45)
    parse.add_argument('--top_k', type=int, default=200)
    args = parse.parse_args()
    return args
        
if __name__ =='__main__':
    args = parse_args()
    eval_model(args)