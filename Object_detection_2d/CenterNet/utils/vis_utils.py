import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from PIL import Image
import cv2
import torch
import torch.nn.functional as F

def denormalize(img_tensor, mean=[0.40789655, 0.44719303, 0.47026116], std=[0.2886383, 0.27408165, 0.27809834]):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return img_tensor * std + mean

def visualize_gt(img_tensor, boxes, labels, class_dict):
    img = denormalize(img_tensor)
    img = img.permute(1, 2, 0).clamp(0, 1).numpy()

    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for i, (box, label) in enumerate(zip(boxes, labels)):
        xmin, ymin, xmax, ymax = box
        # width and height
        w = xmax - xmin
        h = ymax - ymin

        rect = patches.Rectangle(
            (xmin, ymin), w, h, linewidth=2,
            edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

        name = str(class_dict[label])
        ax.text(
            xmin, ymin - 2, name,
            fontsize=10, color='white',
            bbox=dict(facecolor='red', edgecolor='none', alpha=0.7)
        )
    plt.axis("off")
    plt.show()

def visualize_heatmap(hm, cls, img):
    img = denormalize(img)
    img = img.permute(1, 2, 0).clamp(0, 1).numpy()
    img = (img * 255).astype(np.uint8)

    hm = hm[None, ...]
    hm_up = F.interpolate(hm, size=(img.shape[1], img.shape[0]), mode='bilinear', align_corners=False)
    hm_up = hm_up[0, cls].detach().cpu().numpy()
    hm_up = (hm_up * 255).astype(np.uint8)

    hm_color = cv2.applyColorMap(hm_up, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, hm_color, 0.4, 0)
    plt.imshow(overlay[..., ::-1])
    plt.title("Overlay Heatmap")
    plt.axis('off')
    plt.show()

def visualize_wh_offsets(whs, offsets, inds, offsets_mask, img, stride=4, img_size=512):
    img = denormalize(img)
    img = img.permute(1, 2, 0).clamp(0, 1).numpy()
    img = (img * 255).astype(np.uint8)

    Hf, Wf = img_size // stride, img_size // stride
    ys = (inds // Wf).float() + offsets[:, 1]
    xs = (inds % Wf).float() + offsets[:, 0]

    mask = offsets_mask > 0
    xs = xs[mask] * stride
    ys = ys[mask] * stride
    whs = whs[mask] * stride
    ws = whs[:, 0]
    hs = whs[:, 1]

    x1s = (xs - ws/2).cpu().numpy().astype(np.int32)
    y1s = (ys - hs/2).cpu().numpy().astype(np.int32)
    x2s = (xs + ws/2).cpu().numpy().astype(np.int32)
    y2s = (ys + hs/2).cpu().numpy().astype(np.int32)

    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for x1, y1, x2, y2 in zip(x1s, y1s, x2s, y2s):
        w = x2 - x1
        h = y2 - y1

        rect = patches.Rectangle(
            (x1, y1), w, h, linewidth=2,
            edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
    plt.axis("off")
    plt.show()


def visualize_detection(opts, dataset, imgs, detections, idxes, class_dict, save_path, model_name, dataset_type, cmap):
    batch_size = len(detections)
    score_thres = opts.score_thres
    
    colors = cmap(len(class_dict), normalized=True)
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
