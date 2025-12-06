import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import open3d as o3d

class BasePointCloudVisualizer:
    def __init__(self, class_num, model_name, dataset_type,
                 pclouds_num, y_rotate=50, elev=80, azim=-90):
        self.class_num = class_num
        self.model_name = model_name
        self.dataset_type = dataset_type
        self.pclouds_num = pclouds_num
        self.y_rotate = y_rotate
        self.elev = elev
        self.azim = azim

        self.n_rows = 2
        self.n_cols = int(np.ceil(self.pclouds_num / self.n_rows))
        color_maps = self.generate_color_map(class_num)
        self.color_maps = [color_maps.copy() for _ in range(pclouds_num)]

    def visualize(self, pclouds, pred_labels, class_dict):
        raise NotImplementedError

    def rgb_to_hex(self, rgb):
        return '#%02x%02x%02x' % tuple(int(255*x) for x in rgb)

    def generate_color_map(self, class_num):
        cmap = plt.get_cmap('hsv')
        color_map = {}
        for i in range(class_num):
            rgba = cmap(i / class_num)
            rgb = [round(c, 3) for c in rgba[:3]]
            color_map[i] = rgb
        return color_map

    def rotate_points_around_y(self, points, angle_deg):
        angle_rad = np.radians(angle_deg)
        R_y = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])
        return points @ R_y.T

class PointCloudSegVisualizer(BasePointCloudVisualizer):
    def __init__(self, *args, **kwargs):
        is_part_seg = kwargs.pop('is_part_seg', False)
        super().__init__(*args, **kwargs)
        self.is_part_seg = is_part_seg

    def visualize(self, pclouds, pred_labels, class_dict, save_path):
        fig = plt.figure(figsize=(4 * self.n_cols, 4 * self.n_rows))
        for i in range(self.pclouds_num):
            color_map = self.color_maps[i]
            pclouds_np = pclouds[i, :3, :].T.numpy()
            pclouds_np = self.rotate_points_around_y(pclouds_np, self.y_rotate)
        
            colors = np.array([color_map[label] for label in pred_labels[i]])
            ax = fig.add_subplot(self.n_rows, self.n_cols, i + 1, projection='3d')
            ax.scatter(pclouds_np[:, 0], pclouds_np[:, 1], pclouds_np[:, 2], c=colors, s=1)
            ax.view_init(elev=self.elev, azim=self.azim)
            plt.axis('off')

        if not self.is_part_seg:
            legend_elements = []
            for class_id, color in color_map.items():
                legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                label=f'{class_dict[class_id]}',
                                markerfacecolor=self.rgb_to_hex(color),
                                markersize=10))
            fig.legend(handles=legend_elements, loc='lower center', ncol=len(color_map), fontsize='large')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.05)
        fig.suptitle(self.model_name, fontsize=16)
        plt.savefig(os.path.join(save_path, '{}_{}.png'.format(self.model_name, self.dataset_type)), dpi=300, bbox_inches='tight')
        plt.show()

class PointCloudClsVisualizer(BasePointCloudVisualizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def visualize(self, pclouds, pred_labels, class_dict, save_path):
        fig = plt.figure(figsize=(4 * self.n_cols, 4 * self.n_rows))
        for i in range(self.pclouds_num):
            color_map = self.color_maps[i]
            pclouds_np = pclouds[i, :3, :].T.numpy()
            pclouds_np = self.rotate_points_around_y(pclouds_np, self.y_rotate)
        
            colors = np.array([color_map[pred_labels[i]] for _ in range(pclouds_np.shape[0])])
            ax = fig.add_subplot(self.n_rows, self.n_cols, i + 1, projection='3d')
            ax.scatter(pclouds_np[:, 0], pclouds_np[:, 1], pclouds_np[:, 2], c=colors, s=1)
            ax.view_init(elev=self.elev, azim=self.azim)
            ax.set_title(f'{class_dict[predi_labels[i]]}')
            plt.axis('off')
    
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.05)
        fig.suptitle(self.model_name, fontsize=16)
        plt.savefig(os.path.join(save_path, '{}_{}.png'.format(self.model_name, self.dataset_type)), dpi=300, bbox_inches='tight')
        plt.show()