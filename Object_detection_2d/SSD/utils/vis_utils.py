import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from Object_detection_2d.data.dataset.voc import voc_cmap

def visualize_detection(args, imgs, batched_bboxes, batched_labels,
                        batched_scores, class_dict, save_path):
    batch_size = args.test_batch_size
    model_name = args.model
    dataset_type = args.dataset
    year = args.voc_year
    
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
        bboxes = batched_bboxes[i].cpu().numpy() if hasattr(batched_bboxes[i], 'cpu') else batched_bboxes[i]
        labels = batched_labels[i].cpu().numpy() if hasattr(batched_labels[i], 'cpu') else batched_labels[i]
        scores = batched_scores[i].cpu().numpy() if hasattr(batched_scores[i], 'cpu') else batched_scores[i]

        ax.imshow(img)
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
    if dataset_type == "voc":
        save_name = save_name + "_{}".format(year)
    plt.savefig(os.path.join(save_path, save_name + ".png"), bbox_inches='tight')
    plt.show()
