import torch
import argparse
import os
import numpy as np

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from core.utils import is_main_process, init_ddp

from Classification_2d.data import build_dataloader
from Classification_2d.model import build_model
from Classification_2d.utils import build_visualizer
from Classification_2d.utils import parse_config


def test_model(args):
    local_rank, rank, world_size = init_ddp()
    config_path = args.config_path
    exp = args.exp
    opts = parse_config(config_path)

    root = opts.root
    os.makedirs(os.path.join(root, 'runs'), exist_ok=True)
    os.makedirs(os.path.join(root, 'runs', exp), exist_ok=True)
    weight_path = os.path.join(root, 'runs', exp, "max-f1-val.pth")
    save_path = os.path.join(root, 'runs', exp)
    model_name = opts.model.name
    dataset_name = opts.dataset_name
    
    if is_main_process():
        print("Start testing model {}!".format(model_name))

    _, _, test_dataloader = build_dataloader(opts)
    model = build_model(opts.model).to(local_rank)
    model.load_state_dict(torch.load(weight_path, map_location=f"cuda:{local_rank}"))
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model.eval()

    class_dict = test_dataloader.dataset.get_class_dict()
    img_visualizer = build_visualizer(class_dict, opts.visualizer)

    imgs, _ = next(iter(test_dataloader))
    with torch.no_grad():
        outputs = model(imgs.to(local_rank))
        outputs = torch.softmax(outputs, dim=1)
        pred_classes = torch.argmax(outputs, dim=1).cpu().numpy()
        outputs = outputs.cpu().numpy()
        img_visualizer.visualize(imgs, pred_classes, outputs, save_path)

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--exp', type=str, required=True)
    parse.add_argument('--config_path', type=str, required=True)
    args = parse.parse_args()
    return args

if __name__ =='__main__':
    args = parse_args()
    test_model(args)