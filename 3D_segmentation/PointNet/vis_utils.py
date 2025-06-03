import numpy as np
import random
import matplotlib.pyplot as plt

def rotate_points_around_y(points, angle_deg):
    angle_rad = np.radians(angle_deg)
    R_y = np.array([
        [np.cos(angle_rad), 0, np.sin(angle_rad)],
        [0, 1, 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])
    return points @ R_y.T

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(int(255*x) for x in rgb)

def generate_color_map(class_num):
    cmap = plt.get_cmap('hsv')
    color_map = {}
    for i in range(class_num):
        rgba = cmap(i / class_num)
        rgb = [round(c, 3) for c in rgba[:3]]
        color_map[i] = rgb
    return color_map


def get_color_map(args):
    dataset_type = args.dataset
    class_num = args.class_num
    if dataset_type == "chair":
        return {
            0: [1, 0, 0],  # Class 0: Red
            1: [0, 1, 0],  # Class 1: Green
            2: [0, 0, 1],   # Class 2: Blue
            3: [1, 1, 0]
        }
        
    elif dataset_type == "modelnet40":
        return generate_color_map(1)
    
    elif dataset_type == "s3dis":
        return generate_color_map(class_num)
    else:
        raise ValueError(f'Unknown dataset {dataset_type}.')