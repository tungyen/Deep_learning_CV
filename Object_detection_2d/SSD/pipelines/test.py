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

    config_path = args.config
    ckpts_path = args.experiment
    args = parse_config(config_path)
    root = args['root']
    model_name = args['model']
    dataset_type = args['datasets']['name']
    save_path = os.path.join(root, "runs", ckpts_path)
    weight_path = os.path.join(save_path, "{}_{}.pt".format(args['model'], args['datasets']['name']))
    
    if is_main_process():
        print("Start testing model {} on {} dataset!".format(model_name, dataset_type))
    _, _, test_dataloader = build_dataloader(args)
    class_dict = test_dataloader.dataset.class_dict
    model = build_model(args).to(local_rank)
    model.load_state_dict(torch.load(weight_path, map_location=f"cuda:{local_rank}"))
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model.eval()

    imgs, _, idxes = next(iter(test_dataloader))
    mean = args['img_mean']
    imgs_denorm = imgs + torch.tensor(mean).view(1, 3, 1, 1)
    imgs_denorm = imgs_denorm.permute(0, 2, 3, 1).numpy()
    imgs_denorm = imgs_denorm.astype(np.uint8)
    with torch.no_grad():
        detections = model(imgs.to(local_rank), False)
        detections = [d.to(torch.device("cpu")) for d in detections]
    visualize_detection(args, test_dataloader.dataset, imgs_denorm, detections, idxes, class_dict, save_path, model_name, dataset_type)

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--experiment', type=str, required=True)
    parse.add_argument('--config', type=str, required=True)
    args = parse.parse_args()
    return args
    
if __name__ =='__main__':
    args = parse_args()
    test_model(args)