import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
        return generate_color_map(class_num)
    
    elif dataset_type == "s3dis":
        return generate_color_map(class_num)
    else:
        raise ValueError(f'Unknown dataset {dataset_type}.')
    
def visualize_pcloud(args, pcloud, color_map, predict_class, 
                    class_dict=None, y_rotate=50, elev=80, azim=-90):
    n_rows = 2
    n_cols = int(np.ceil(args.batch_size / n_rows))
    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    task = args.model[-3:]
        
    for i in range(args.batch_size):
        points_np = pcloud[i].T.numpy()
        points_np = rotate_points_around_y(points_np, y_rotate)
        
        if task == "seg":
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

    if task == "seg":
        legend_elements = []
        for class_id, color in color_map.items():
            legend_elements.append(Line2D([0], [0], marker='o', color='w',
                            label=f'{class_dict[class_id]}',
                            markerfacecolor=rgb_to_hex(color),
                            markersize=10))
        fig.legend(handles=legend_elements, loc='lower center', ncol=len(color_map), fontsize='large')
        
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)
    plt.savefig('img/{}_{}.png'.format(args.model, args.dataset), dpi=300, bbox_inches='tight')
    plt.show()