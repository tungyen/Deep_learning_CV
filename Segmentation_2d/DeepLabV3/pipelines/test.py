import torch
import argparse
import os
import numpy as np

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from Segmentation_2d.dataset.utils import get_dataset
from Segmentation_2d.utils import get_model, setup_args_with_dataset
from Segmentation_2d.vis_utils import visualize_image_seg


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
    ckpts_path = args.experiment
    
    args = setup_args_with_dataset(dataset_type, args)
    
    if dataset_type == 'cityscapes':
        weight_path = os.path.join(root, ckpts_path, "{}_{}.pth".format(model_name, dataset_type))
    elif dataset_type == 'voc':
        weight_path = os.path.join(root, ckpts_path, "{}_{}_{}.pth".format(model_name, dataset_type, args.voc_year))
    else:
        raise ValueError(f'Unknown dataset {dataset_type}.')
    if dist.get_rank() == 0:
        print("Start testing model {} on {} dataset!".format(model_name, dataset_type))
    model = get_model(args).to(local_rank)
    model.load_state_dict(torch.load(weight_path, map_location=f"cuda:{local_rank}"))
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model.eval()

    _, _, test_dataloader, _, mean, std = get_dataset(args)
    imgs, _ = next(iter(test_dataloader))
    imgs_denorm = imgs * std + mean
    imgs_denorm = imgs_denorm.permute(0, 2, 3, 1).numpy()
    imgs_denorm = (imgs_denorm * 255).astype(np.uint8)
    with torch.no_grad():
        outputs = model(imgs.to(local_rank))
        predict_class = torch.argmax(outputs, dim=1).cpu().numpy()
        visualize_image_seg(args, predict_class, imgs_denorm, save_path)

def parse_args():
    parse = argparse.ArgumentParser()
    # Dataset
    parse.add_argument('--dataset', type=str, default="cityscapes")
    parse.add_argument('--crop_size', type=list, default=[512, 512])
    parse.add_argument('--voc_data_root', type=str, default="Dataset/VOC")
    parse.add_argument('--voc_year', type=str, default="2012_aug")
    parse.add_argument('--voc_download', type=bool, default=False)
    parse.add_argument('--voc_crop_val', type=bool, default=True)
    parse.add_argument('--cityscapes_crop_val', type=bool, default=False)
    
    # Model
    parse.add_argument('--model', type=str, default="deeplabv3")
    parse.add_argument('--backbone', type=str, default="resnet101")
    parse.add_argument('--bn_momentum', type=float, default=0.1)
    
    # Training
    parse.add_argument('--experiment', type=str, required=True)
    args = parse.parse_args()
    return args

if __name__ =='__main__':
    args = parse_args()
    test_model(args)