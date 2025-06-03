import os
import torch
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    
    weight_path = "ckpts/{}_{}.pth".format(model_name, dataset_type)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model = model.to(device)
    model.eval()

    _, _, test_dataloader, class_dict = get_dataset(args)
    test_size = len(test_dataloader.dataset)
    
    if dataset_type == "chair":
        n_cols = 3
        n_rows = int(np.ceil(test_size / n_cols))
        fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
        
        for idx, pcloud in enumerate(test_dataloader):
            with torch.no_grad():
                pcloud = pcloud.to(device).float()
                output = torch.squeeze(model(pcloud))
                predict = torch.softmax(output, dim=0).cpu()
                predict_cla = torch.argmax(predict, dim=0).numpy()
                
                points_np = torch.squeeze(pcloud).T.cpu().numpy()
                points_np = rotate_points_around_y(points_np, 50)
                colors = np.array([color_map[label] for label in predict_cla])
                
                ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
                ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=colors, s=1)
                ax.view_init(elev=80, azim=-90)
                ax.set_title(f'Sample {idx}')
                plt.axis('off')

        legend_elements = []
        for class_id, color in color_map.items():
            legend_elements.append(Line2D([0], [0], marker='o', color='w',
                            label=f'Class {class_id}',
                            markerfacecolor=rgb_to_hex(color),
                            markersize=10))

        fig.legend(handles=legend_elements, loc='lower center', ncol=len(color_map), fontsize='large')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.05)
        plt.savefig('img/{}_{}.png'.format(model_name, dataset_type), dpi=300, bbox_inches='tight')
        plt.show()
        
    elif dataset_type == "modelnet40":
        pass
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
    parse.add_argument('--batch_size', type=int, default=1)
    parse.add_argument('--device', type=str, default="cuda")
    args = parse.parse_args()
    return args
                
                
if __name__ =='__main__':
    args = parse_args()
    test_model(args)