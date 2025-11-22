from tqdm import tqdm
import argparse
import numpy as np
import os
import torch
import torch.optim as optim

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from Object_detection_2d.data import build_dataloader
from Object_detection_2d.SSD.model import build_model
from Object_detection_2d.SSD.utils.config_utils import parse_config
from Object_detection_2d.SSD.utils.ddp_utils import is_main_process
from Object_detection_2d.SSD.utils.vis_utils import visualize_detection

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
    weight_path = os.path.join(root, 'runs', exp, "max-ap-val.pt")
    save_path = os.path.join(root, 'runs', exp)

    model_name = opts.model.name
    dataset_type = opts.datasets.dataset_name
    
    if is_main_process():
        print("Start testing model {} on {} dataset!".format(model_name, dataset_type))
    _, _, test_dataloader = build_dataloader(opts)
    class_dict = test_dataloader.dataset.class_dict
    model = build_model(opts.model).to(local_rank)
    model.load_state_dict(torch.load(weight_path, map_location=f"cuda:{local_rank}"))
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model.eval()

    imgs, _, idxes = next(iter(test_dataloader))
    mean = opts.img_mean
    imgs_denorm = imgs + torch.tensor(mean).view(1, 3, 1, 1)
    imgs_denorm = imgs_denorm.permute(0, 2, 3, 1).numpy()
    imgs_denorm = imgs_denorm.astype(np.uint8)
    with torch.no_grad():
        detections = model(imgs.to(local_rank), False)
        detections = [d.to(torch.device("cpu")) for d in detections]
    visualize_detection(opts, test_dataloader.dataset, imgs_denorm, detections, idxes, class_dict, save_path, model_name, dataset_type)

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--exp', type=str, required=True)
    parse.add_argument('--config_path', type=str, required=True)
    args = parse.parse_args()
    return args
    
if __name__ =='__main__':
    args = parse_args()
    test_model(args)