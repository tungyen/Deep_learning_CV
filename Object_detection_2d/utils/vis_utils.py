import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from PIL import Image
import cv2
import torch
import torch.nn.functional as F

class ImageDetectionVisualizer:
    def __init__(self,
                 class_dict,
                 cmap,
                 mean,
                 std=None,
                 hm_alpha=0.6,
                 score_thres=0.7,
                 stride=None,
                 img_size=None):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1) if std is not None else None
        self.class_dict = class_dict
        self.cmap = cmap
        self.hm_alpha = hm_alpha
        self.stride = stride
        self.img_size = img_size
        self.score_thres = score_thres

    def denormalize(self, img_tensor):
        img_tensor = img_tensor * self.std + self.mean if self.std is not None else img_tensor + self.mean
        img_tensor = img_tensor.permute(0, 2, 3, 1).numpy()
        return img_tensor

    def visualize_gt(img_tensor, boxes, labels):
        img = denormalize(img_tensor)
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

            name = str(self.class_dict[label])
            ax.text(
                xmin, ymin - 2, name,
                fontsize=10, color='white',
                bbox=dict(facecolor='red', edgecolor='none', alpha=0.7)
            )
        plt.axis("off")
        plt.show()

    def visualize_heatmap(self, hm, cls, img):
        img = denormalize(img)
        img = (img * 255).astype(np.uint8)

        hm = hm[None, ...]
        hm_up = F.interpolate(hm, size=(img.shape[1], img.shape[0]), mode='bilinear', align_corners=False)
        hm_up = hm_up[0, cls].detach().cpu().numpy()
        hm_up = (hm_up * 255).astype(np.uint8)

        hm_color = cv2.applyColorMap(hm_up, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, self.hm_alpha, hm_color, 1-self.hm_alpha, 0)
        plt.imshow(overlay[..., ::-1])
        plt.title("Overlay Heatmap")
        plt.axis('off')
        plt.show()

    def visualize_wh_offsets(self, whs, offsets, inds, offsets_mask, img):
        assert self.stride is not None, "Stride should not be None."
        assert self.img_size is not None, "Image Size should not be None."
        
        img = denormalize(img)

        Hf, Wf = self.img_size // self.stride, self.img_size // self.stride
        ys = (inds // Wf).float() + offsets[:, 1]
        xs = (inds % Wf).float() + offsets[:, 0]

        mask = offsets_mask > 0
        xs = xs[mask] * self.stride
        ys = ys[mask] * self.stride
        whs = whs[mask] * self.stride
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

    def visualize_detection(self, input_dict: dict, detections, img_info, save_path):
        batch_size = len(detections)
        imgs = self.denormalize(input_dict['img'])
        if np.max(imgs) <= 1.0:
            imgs = imgs * 255.0 
        imgs = imgs.astype(np.uint8)
        scale = input_dict['scale']
        padding = input_dict['padding']
        rescale_size = input_dict['rescale_size']
        
        colors = self.cmap(len(self.class_dict), normalized=True)
        row = 2
        col = batch_size // row

        fig, axes = plt.subplots(row, col, figsize=(8, 8))
        axes = axes.flatten() if row > 1 else [axes]
        for i, ax in enumerate(axes):
            if i >= batch_size:
                ax.axis('off')
                continue

            img = imgs[i] # (H, W, C)
            pad_w, pad_h = padding[i, 0], padding[i, 1]
            rescale_w, rescale_h = rescale_size[i, 0], rescale_size[i, 1]
            img = img[pad_h:pad_h+rescale_h, pad_w:pad_w+rescale_w, :]
            detections[i] = detections[i].rescale(scale[i], padding[i]).numpy()
            bboxes = detections[i]['boxes']
            labels = detections[i]['labels']
            scores = detections[i]['scores']

            mask = scores >= self.score_thres
            bboxes = bboxes[mask]
            labels = labels[mask]
            scores = scores[mask]

            img_resized = Image.fromarray(img).resize((img_info[i]['width'], img_info[i]['height']), Image.Resampling.LANCZOS)
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
                class_name = self.class_dict[label]
                ax.text(
                    xmin, ymin - 5,
                    f"{class_name} {score:.2f}",
                    fontsize=10,
                    color='white',
                    bbox=dict(facecolor=color, alpha=0.7, pad=2, edgecolor='none')
                )
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "det_result.png"), bbox_inches='tight')
        plt.show()
