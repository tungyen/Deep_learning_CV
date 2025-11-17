import torch
import argparse
import os
import numpy as np

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from Segmentation_2d.data import build_dataloader
from Segmentation_2d.DeepLabV3.model import build_model
from Segmentation_2d.utils import visualize_image_seg, is_main_process, parse_config


def test_model(args):
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
    weight_path = os.path.join(root, 'runs', exp, "max-iou-val.pth")
    save_path = os.path.join(root, 'runs', exp)
    model_name = opts.model.name
    dataset_name = opts.dataset_name
    bs = opts.test_batch_size
    
    if is_main_process():
        print("Start testing model {}!".format(model_name))

    _, _, test_dataloader = build_dataloader(opts)
    model = build_model(opts.model).to(local_rank)
    model.load_state_dict(torch.load(weight_path, map_location=f"cuda:{local_rank}"))
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model.eval()

    imgs, _ = next(iter(test_dataloader))
    std = torch.tensor(opts.img_std).view(1, 3, 1, 1)
    mean = torch.tensor(opts.img_mean).view(1, 3, 1, 1)
    imgs_denorm = imgs * std + mean
    imgs_denorm = imgs_denorm.permute(0, 2, 3, 1).numpy()
    imgs_denorm = (imgs_denorm * 255).astype(np.uint8)
    with torch.no_grad():
        outputs = model(imgs.to(local_rank))
        predict_class = torch.argmax(outputs, dim=1).cpu().numpy()
        visualize_image_seg(dataset_name, bs, predict_class, imgs_denorm, save_path)

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--exp', type=str, required=True)
    parse.add_argument('--config_path', type=str, required=True)
    args = parse.parse_args()
    return args

if __name__ =='__main__':
    args = parse_args()
    test_model(args)