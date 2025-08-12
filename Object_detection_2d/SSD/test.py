import torch
import argparse
import os
import numpy as np

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from Object_detection_2d.dataset.utils import get_dataset
from Object_detection_2d.utils import (
    get_model,
    setup_args_with_dataset,
    decode_boxes,
    gather_list_ddp
)

def test_model(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    root = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(root, args.experiment)
    os.makedirs(save_path, exist_ok=True)
    model_name = args.model
    dataset_type = args.dataset
    args = setup_args_with_dataset(dataset_type, args)
    
    if dist.get_rank() == 0:
        print("Start testing model {} on {} dataset!".format(model_name, dataset_type))
    
    ckpts_path = args.experiment
    weight_path = os.path.join(root, ckpts_path, "{}_{}.pth".format(model_name, dataset_type))
    
    model = get_model(args).to(local_rank)
    model.load_state_dict(torch.load(weight_path, map_location=f"cuda:{local_rank}"))
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model.eval()

    _, _, test_dataloader, class_dict, mean, std = get_dataset(args)

    imgs, targets = next(iter(test_dataloader))
    imgs_denorm = imgs * std + mean
    imgs_denorm = imgs_denorm.permute(0, 2, 3, 1).numpy()
    imgs_denorm = (imgs_denorm * 255).astype(np.uint8)
    with torch.no_grad():
        pred_boxes, pred_scores = model(imgs.to(local_rank))
        pred_boxes_batch, pred_labels_batch, pred_scores_batch = decode_boxes(
            args, pred_boxes, pred_scores,
            model.module.prior_boxes_center
        )
        


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
    
    # testing
    parse.add_argument('--experiment', type=str, required=True)
    parse.add_argument('--min_scores', type=float, default=0.01)
    parse.add_argument('--max_overlap', type=float, default=0.45)
    parse.add_argument('--top_k', type=int, default=200)
    args = parse.parse_args()
    return args
    
if __name__ =='__main__':
    args = parse_args()
    test_model(args)