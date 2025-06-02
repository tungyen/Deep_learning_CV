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

color_map = {
    0: [1, 0, 0],  # Class 0: Red
    1: [0, 1, 0],  # Class 1: Green
    2: [0, 0, 1],   # Class 2: Blue
    3: [1, 1, 0]
}

def test_model(args):
    device = args.device
    model_name = args.model
    dataset_type = args.dataset

    train_dataloader, val_dataloader, _, val_num = get_dataset(args)
    model = get_model(args)
    
    weight_path = "ckpts/{}_{}.pth".format(model_name, dataset_type)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model = model.to(device)
    model.eval()

    _, _, test_dataloader, _ = get_dataset(args)
    
    for pcloud in test_dataloader:
        with torch.no_grad():
            pcloud = pcloud.to(device).float()
            output = torch.squeeze(model(pcloud))
            predict = torch.softmax(output, dim=0).cpu()
            predict_cla = torch.argmax(predict, dim=0).numpy()
            
            point_cloud = o3d.geometry.PointCloud()
            points_np = torch.squeeze(pcloud).T.cpu().numpy()

            point_cloud.points = o3d.utility.Vector3dVector(points_np)
            colors = np.array([color_map[label] for label in predict_cla])
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
            # o3d.visualization.draw_geometries([point_cloud])

            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=colors, s=1)
            ax.view_init(elev=40, azim=-90)
            plt.axis('off')

            def rgb_to_hex(rgb):
                return '#%02x%02x%02x' % tuple(int(255*x) for x in rgb)

            legend_elements = []
            for class_id, color in color_map.items():
                legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                  label=f'Class {class_id}',
                                  markerfacecolor=rgb_to_hex(color),
                                  markersize=10))


            ax.legend(handles=legend_elements, loc='upper right', fontsize='large')
            plt.savefig('img/{}_{}.png'.format(model_name, dataset_type), dpi=300, bbox_inches='tight')
            plt.show()


def parse_args():
    parse = argparse.ArgumentParser()
    # Dataset
    parse.add_argument('--dataset', type=str, default="chair")
    parse.add_argument('--data_path', type=str, default="../../Dataset/Chair_dataset")
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