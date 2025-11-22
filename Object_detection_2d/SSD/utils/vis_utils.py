import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from PIL import Image

from Object_detection_2d.data.dataset.voc import voc_cmap

def visualize_detection(opts, dataset, imgs, detections, idxes, class_dict, save_path, model_name, dataset_type):
    batch_size = len(detections)
    score_thres = opts.score_thres
    
    colors = voc_cmap(len(class_dict), normalized=True)
    row = 2
    col = batch_size // row

    fig, axes = plt.subplots(row, col, figsize=(8, 8))
    axes = axes.flatten() if row > 1 else [axes]
    for i, ax in enumerate(axes):
        if i >= batch_size:
            ax.axis('off')
            continue

        img = imgs[i] # (H, W, C)
        img_info = dataset.get_img_info(idxes[i])
        detections[i] = detections[i].resize((img_info['width'], img_info['height'])).numpy()
        bboxes = detections[i]['boxes']
        labels = detections[i]['labels']
        scores = detections[i]['scores']

        mask = scores >= score_thres
        bboxes = bboxes[mask]
        labels = labels[mask]
        scores = scores[mask]

        img_resized = Image.fromarray(img).resize((img_info['width'], img_info['height']), Image.Resampling.LANCZOS)

        ax.imshow(img_resized)
        for box, label, score in zip(bboxes, labels, scores):
            xmin, ymin, xmax, ymax = box
            color = colors[label]
            rect = patches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)
            class_name = class_dict[label]
            ax.text(
                xmin, ymin - 5,
                f"{class_name} {score:.2f}",
                fontsize=10,
                color='white',
                bbox=dict(facecolor=color, alpha=0.7, pad=2, edgecolor='none')
            )
        ax.axis('off')
    
    plt.tight_layout()
    save_name = '{}_{}'.format(model_name, dataset_type)
    plt.savefig(os.path.join(save_path, save_name + ".png"), bbox_inches='tight')
    plt.show()
