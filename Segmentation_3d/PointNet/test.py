import torch
import argparse
import os

from Segmentation_3d.dataset.utils import get_dataset
from Segmentation_3d.utils import get_model, setup_args_with_dataset
from Segmentation_3d.vis_utils import visualize_pcloud, get_color_map

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
    ckpts_path = "ckpts"
    weight_path = os.path.join(root, ckpts_path, "{}_{}_{}.pth".format(model_name, dataset_type, task))
    
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model = model.to(device)
    model.eval()

    _, _, test_dataloader, class_dict = get_dataset(args)
    
    if dataset_type == "chair":
        pclouds = next(iter(test_dataloader))
        with torch.no_grad():
            outputs, _ = model(pclouds.to(device).float())
            predict_classes = torch.argmax(outputs, dim=1).cpu().numpy()
        visualize_pcloud(args, pclouds, color_map, predict_classes, save_path, class_dict)
    elif dataset_type == "modelnet40":
        pclouds, _ = next(iter(test_dataloader))
        with torch.no_grad():
            outputs, _ = model(pclouds.to(device))
            predict_classes = torch.argmax(outputs, dim=1).cpu().numpy()    
        visualize_pcloud(args, pclouds, color_map, predict_classes, save_path, class_dict)
    elif dataset_type == "s3dis":
        pass
    else:
        raise ValueError(f'Unknown dataset {dataset_type}.')


def parse_args():
    parse = argparse.ArgumentParser()
    # Dataset
    parse.add_argument('--dataset', type=str, default="chair")
    
    # Model
    parse.add_argument('--model', type=str, default="pointnet_plus_ssg")
    
    # testing
    parse.add_argument('--batch_size', type=int, default=6)
    parse.add_argument('--device', type=str, default="cuda")
    args = parse.parse_args()
    return args
    
if __name__ =='__main__':
    args = parse_args()
    test_model(args)