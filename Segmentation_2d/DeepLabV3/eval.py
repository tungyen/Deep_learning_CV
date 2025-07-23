import torch
from tqdm import tqdm
import argparse
import os

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from Segmentation_2d.dataset.utils import get_dataset
from Segmentation_2d.utils import get_model, setup_args_with_dataset
from Segmentation_2d.metrics import ConfusionMatrix


def eval_model(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    model_name = args.model
    dataset_type = args.dataset
    root = os.path.dirname(os.path.abspath(__file__))
    args = setup_args_with_dataset(dataset_type, args)
    ckpts_path = args.experiment
    
    if dataset_type == 'cityscapes':
        weight_path = os.path.join(root, ckpts_path, "{}_{}.pth".format(model_name, dataset_type))
    elif dataset_type == 'voc':
        weight_path = os.path.join(root, ckpts_path, "{}_{}_{}.pth".format(model_name, dataset_type, args.voc_year))
    else:
        raise ValueError(f'Unknown dataset {dataset_type}.')
    
    _, val_dataloader, _, class_dict, _, _ = get_dataset(args)
    model = get_model(args).to(local_rank)
    model.load_state_dict(torch.load(weight_path, map_location=f"cuda:{local_rank}"))
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model.eval()
    
    confusion_matrix = ConfusionMatrix(class_num=args.class_num, ignore_index=args.ignore_idx)
    if dist.get_rank() == 0:
        print("Start evaluation model {} on {} dataset!".format(model_name, dataset_type))
    
    with torch.no_grad():
        for imgs, labels in tqdm(val_dataloader, desc="Evaluate", disable=dist.get_rank() != 0):
            output = model(imgs.to(local_rank))
            pred_class = torch.argmax(output, dim=1)
            confusion_matrix.update(pred_class.cpu(), labels)
    
    if dist.get_rank() == 0:
        metrics = confusion_matrix.compute_metrics()
        print("Validation mIoU of {} on {} ===>{:.4f}".format(model_name, dataset_type, metrics['mious'].item()))
        for i, iou in enumerate(metrics['ious']):
            print("{} IoU: {:.4f}".format(class_dict[i], iou))
    dist.destroy_process_group()

def parse_args():
    parse = argparse.ArgumentParser()
    # Dataset
    parse.add_argument('--dataset', type=str, default="cityscapes")
    parse.add_argument('--crop_size', type=list, default=[512, 1024])
    parse.add_argument('--voc_data_root', type=str, default="Dataset/VOC")
    parse.add_argument('--voc_year', type=str, default="2012_aug")
    parse.add_argument('--voc_download', type=bool, default=False)
    parse.add_argument('--voc_crop_val', type=bool, default=True)
    parse.add_argument('--cityscapes_crop_val', type=bool, default=True)
    
    # Model
    parse.add_argument('--model', type=str, default="deeplabv3")
    parse.add_argument('--backbone', type=str, default="resnet101")
    parse.add_argument('--bn_momentum', type=float, default=0.1)
    
    # Validation
    parse.add_argument('--experiment', type=str, required=True)
    args = parse.parse_args()
    return args

if __name__ =='__main__':
    args = parse_args()
    eval_model(args)