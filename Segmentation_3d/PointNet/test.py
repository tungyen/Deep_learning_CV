import torch
import argparse

from dataset import *
from model import *
from utils import *
from vis_utils import *


def test_model(args):
    device = args.device
    model_name = args.model
    dataset_type = args.dataset

    model = get_model(args)
    color_map = get_color_map(args)
    
    print("Start testing model {} on {} dataset!".format(model_name, dataset_type))
    
    weight_path = "ckpts/{}_{}.pth".format(model_name, dataset_type)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model = model.to(device)
    model.eval()

    _, _, test_dataloader, class_dict = get_dataset(args)
    
    if dataset_type == "chair":
        for pclouds in test_dataloader:
            with torch.no_grad():
                outputs = torch.squeeze(model(pclouds.to(device).float()))
                predict_classes = torch.argmax(outputs, dim=1).numpy()
                
            visualize_pcloud(args, pclouds, color_map, predict_classes, class_dict)
        
    elif dataset_type == "modelnet40":
        for pclouds, _ in test_dataloader:
            with torch.no_grad():
                outputs = model(pclouds.to(device))
                predict_classes = torch.argmax(outputs, dim=1).cpu().numpy()
                
            visualize_pcloud(args, pclouds, color_map, predict_classes, class_dict)
            break
        
    elif dataset_type == "s3dis":
        pass
    else:
        raise ValueError(f'Unknown dataset {dataset_type}.')


def parse_args():
    parse = argparse.ArgumentParser()
    # Dataset
    parse.add_argument('--dataset', type=str, default="chair")
    parse.add_argument('--n_points', type=int, default=1500)
    
    # Model
    parse.add_argument('--model', type=str, default="pointnet_seg")
    parse.add_argument('--class_num', type=int, default=4)
    
    # testing
    parse.add_argument('--batch_size', type=int, default=6)
    parse.add_argument('--device', type=str, default="cuda")
    args = parse.parse_args()
    return args
                
                
if __name__ =='__main__':
    args = parse_args()
    test_model(args)