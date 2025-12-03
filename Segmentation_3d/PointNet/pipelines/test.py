import torch
import argparse
import os
import numpy as np

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from Segmentation_3d.dataset.utils import get_dataset
from Segmentation_3d.utils import get_model, setup_args_with_dataset
from Segmentation_3d.vis_utils import visualize_pcloud, get_color_map, generate_color_map

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
    color_map = get_color_map(args)
    if dist.get_rank() == 0:
        print("Start testing model {} on {} dataset!".format(model_name, dataset_type))
    
    task = args.task
    ckpts_path = args.experiment
    weight_path = os.path.join(root, ckpts_path, "{}_{}_{}.pth".format(model_name, dataset_type, task))
    
    model = get_model(args).to(local_rank)
    model.load_state_dict(torch.load(weight_path, map_location=f"cuda:{local_rank}"))
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model.eval()

    _, _, test_dataloader, class_dict = get_dataset(args)
    
    if task in ["cls", "semseg"]:
        if dataset_type == "chair":
            pclouds = next(iter(test_dataloader))
            with torch.no_grad():
                outputs, _ = model(pclouds.to(local_rank).float())
                predict_classes = torch.argmax(outputs, dim=1).cpu().numpy()
            visualize_pcloud(args, pclouds, color_map, predict_classes, save_path, class_dict)
        else:
            pclouds, _ = next(iter(test_dataloader))
            with torch.no_grad():
                outputs, _ = model(pclouds.to(local_rank))
                print("Shape of outputs:", outputs.shape)
                predict_classes = torch.argmax(outputs, dim=1).cpu().numpy()    
            visualize_pcloud(args, pclouds, color_map, predict_classes, save_path, class_dict)
    elif task == "partseg":
        pclouds, cls_labels, _ = next(iter(test_dataloader))
        instance2parts, _, label2class = class_dict
        color_map = []
        with torch.no_grad():
            outputs, _ = model(pclouds.to(local_rank), cls_labels.to(local_rank))
            outputs = outputs.cpu().numpy()
            predict_classes = np.zeros((outputs.shape[0], outputs.shape[2])).astype(np.int32)
            for i in range(outputs.shape[0]):
                cls = label2class[cls_labels[i].item()]
                parts_len = len(instance2parts[cls])
                color_map.append(generate_color_map(parts_len))
                logits = outputs[i, :, :]
                predict_classes[i, :] = np.argmax(logits[instance2parts[cls], :], 0)
        visualize_pcloud(args, pclouds, color_map, predict_classes, save_path, class_dict)
    else:
        raise ValueError(f'Unknown dataset {dataset_type}.')


def parse_args():
    parse = argparse.ArgumentParser()
    # Dataset
    parse.add_argument('--dataset', type=str, default="shapenet")
    
    # S3DIS
    parse.add_argument('--test_area', type=int, default=5)
    parse.add_argument('--max_dropout', type=float, default=0.95)
    parse.add_argument('--block_type', type=str, default='dynamic')
    parse.add_argument('--block_size', type=float, default=1.0)
    
    # ShapeNet
    parse.add_argument('--normal_channel', type=bool, default=True)
    parse.add_argument('--class_choice', type=list, default=None)
    
    # Model
    parse.add_argument('--model', type=str, default="pointnet")
    
    # testing
    parse.add_argument('--experiment', type=str, required=True)
    args = parse.parse_args()
    return args
    
if __name__ =='__main__':
    args = parse_args()
    test_model(args)