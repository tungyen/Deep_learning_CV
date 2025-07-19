import torch
import argparse
import os
import numpy as np

from Segmentation_3d.dataset.utils import get_dataset
from Segmentation_3d.utils import get_model, setup_args_with_dataset
from Segmentation_3d.vis_utils import visualize_pcloud, get_color_map, generate_color_map

def test_model(args):
    root = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(root, "imgs")
    os.makedirs(save_path, exist_ok=True)
    device = args.device
    model_name = args.model
    dataset_type = args.dataset
    
    args = setup_args_with_dataset(dataset_type, args)
    model = get_model(args)
    color_map = get_color_map(args)
    
    print("Start testing model {} on {} dataset!".format(model_name, dataset_type))
    
    task = args.task
    ckpts_path = args.experiment
    weight_path = os.path.join(root, ckpts_path, "{}_{}_{}.pth".format(model_name, dataset_type, task))
    
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model = model.to(device)
    model.eval()

    _, _, test_dataloader, class_dict = get_dataset(args)
    
    if task in ["cls", "semseg"]:
        if dataset_type == "chair":
            pclouds = next(iter(test_dataloader))
            with torch.no_grad():
                outputs, _ = model(pclouds.to(device).float())
                predict_classes = torch.argmax(outputs, dim=1).cpu().numpy()
            visualize_pcloud(args, pclouds, color_map, predict_classes, save_path, class_dict)
        else:
            pclouds, _ = next(iter(test_dataloader))
            with torch.no_grad():
                outputs, _ = model(pclouds.to(device))
                predict_classes = torch.argmax(outputs, dim=1).cpu().numpy()    
            visualize_pcloud(args, pclouds, color_map, predict_classes, save_path, class_dict)
    elif task == "partseg":
        pclouds, cls_labels, _ = next(iter(test_dataloader))
        instance2parts, _, label2class = class_dict
        color_map = []
        with torch.no_grad():
            outputs, _ = model(pclouds.to(device), cls_labels.to(device))
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
    parse.add_argument('--block_type', type=str, default='static')
    parse.add_argument('--block_size', type=float, default=1.0)
    
    # ShapeNet
    parse.add_argument('--normal_channel', type=bool, default=True)
    parse.add_argument('--class_choice', type=list, default=None)
    
    # Model
    parse.add_argument('--model', type=str, default="pointnet")
    
    # testing
    parse.add_argument('--experiment', type=str, required=True)
    parse.add_argument('--device', type=str, default="cuda")
    args = parse.parse_args()
    return args
    
if __name__ =='__main__':
    args = parse_args()
    test_model(args)