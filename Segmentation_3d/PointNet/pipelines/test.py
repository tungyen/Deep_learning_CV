import torch
import argparse
import os
import numpy as np

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from core.utils import is_main_process, init_ddp

from Segmentation_3d.data import build_dataloader
from Segmentation_3d.PointNet.model import build_model
from Segmentation_3d.utils import build_visualizer
from Segmentation_3d.utils import parse_config

def test_model(args):
    local_rank, rank, world_size = init_ddp()
    config_path = args.config_path
    exp = args.exp
    opts = parse_config(config_path)

    root = opts.root
    os.makedirs(os.path.join(root, 'runs'), exist_ok=True)
    os.makedirs(os.path.join(root, 'runs', exp), exist_ok=True)
    weight_path = os.path.join(root, 'runs', exp, "max-iou-val.pth")
    save_path = os.path.join(root, 'runs', exp)

    if is_main_process():
        print("Start testing model {}!".format(opts.model.name))

    model_name = opts.model.name
    dataset_type = opts.dataset_name
    
    model = build_model(opts.model).to(local_rank)
    model.load_state_dict(torch.load(weight_path, map_location=f"cuda:{local_rank}"))
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model.eval()

    _, _, test_dataloader = build_dataloader(opts)
    class_dict = test_dataloader.dataset.get_class_dict()
    pclouds_visualizer = build_visualizer(opts.visualizer)

    inputs = next(iter(test_dataloader))
    with torch.no_grad():
        if not isinstance(inputs, list):
            pclouds = inputs
            outputs, _ = model(pclouds.to(local_rank).float())
            pred_labels = torch.argmax(outputs, dim=1).cpu().numpy()
        else:
            pclouds = inputs[0]
            cls_labels = inputs[1]
            outputs, _ = model(pclouds.to(local_rank), cls_labels.to(local_rank))
            pred_labels = model.module.post_process(outputs, cls_labels, class_dict).cpu().numpy()
    pclouds_visualizer.visualize(pclouds, pred_labels, class_dict, save_path)

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--exp', type=str, required=True)
    parse.add_argument('--config_path', type=str, required=True)
    args = parse.parse_args()
    return args
    
if __name__ =='__main__':
    args = parse_args()
    test_model(args)