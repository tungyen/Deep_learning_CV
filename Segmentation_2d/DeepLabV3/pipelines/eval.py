import torch
from tqdm import tqdm
import argparse
import os

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from core.metrics import build_metrics
from core.utils import is_main_process

from Segmentation_2d.data import build_dataloader
from Segmentation_2d.DeepLabV3.model import build_model
from Segmentation_2d.utils import parse_config

def eval_model(args):
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
    
    model_name = opts.model.name
    _, val_dataloader, _ = build_dataloader(opts)
    model = build_model(opts.model).to(local_rank)
    model.load_state_dict(torch.load(weight_path, map_location=f"cuda:{local_rank}"))
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model.eval()
    class_dict = val_dataloader.dataset.get_class_dict()
    metrics = build_metrics(class_dict, val_dataloader.dataset, opts.metrics)
    
    if is_main_process():
        print("Start evaluation model {}!".format(model_name))
    
    with torch.no_grad():
        for imgs, labels in tqdm(val_dataloader, desc="Evaluate", disable=not is_main_process()):
            output = model(imgs.to(local_rank))
            pred_class = torch.argmax(output, dim=1)
            metrics.update(pred_class.cpu(), labels)
    
    metrics.gather(local_rank)
    if is_main_process():
        metrics_result = metrics.compute_metrics()
    dist.destroy_process_group()

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--exp', type=str, required=True)
    parse.add_argument('--config_path', type=str, required=True)
    args = parse.parse_args()
    return args

if __name__ =='__main__':
    args = parse_args()
    eval_model(args)