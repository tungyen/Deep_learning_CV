import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

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
    task = args.task
    if task == "cls":
        class_num = args.cls_class_num
    elif task in ["semseg", "partseg"]:
        class_num = args.seg_class_num
    else:
        raise ValueError(f'Unknown task {task}.')
    return generate_color_map(class_num)
    
def visualize_pcloud(args, pcloud, color_maps, predict_class, save_path,
                    class_dict=None, y_rotate=50, elev=80, azim=-90):
    n_rows = 2
    n_cols = int(np.ceil(args.test_batch_size / n_rows))
    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    task = args.task
    if not isinstance(color_maps, list):
        color_maps = [color_maps.copy() for _ in range(args.test_batch_size)]
    for i in range(args.test_batch_size):
        color_map = color_maps[i]
        points_np = pcloud[i, :3, :].T.numpy()
        points_np = rotate_points_around_y(points_np, y_rotate)
        
        if task in ["semseg", "partseg"]:
            colors = np.array([color_map[label] for label in predict_class[i]])
        elif task == "cls":
            colors = np.array([color_map[predict_class[i]] for _ in range(args.n_points)])
        else:
            raise ValueError(f'Unknown task {task}')
        
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
        ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=colors, s=1)
        ax.view_init(elev=elev, azim=azim)
        
        if task == "cls" and class_dict is not None:
            ax.set_title(f'{class_dict[predict_class[i]]}')
        plt.axis('off')

    if task == "semseg":
        legend_elements = []
        for class_id, color in color_map.items():
            legend_elements.append(Line2D([0], [0], marker='o', color='w',
                            label=f'{class_dict[class_id]}',
                            markerfacecolor=rgb_to_hex(color),
                            markersize=10))
        fig.legend(handles=legend_elements, loc='lower center', ncol=len(color_map), fontsize='large')
    
    model2title = {"pointnet": "PointNet", "pointnet_plus_ssg": "PointNet++ SSG", "pointnet_plus_msg": "PointNet++ MSG"} 
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.05)
    fig.suptitle(model2title[args.model], fontsize=16)
    plt.savefig(os.path.join(save_path, '{}_{}_{}2.png'.format(args.model, args.dataset, task)), dpi=300, bbox_inches='tight')
    plt.show()