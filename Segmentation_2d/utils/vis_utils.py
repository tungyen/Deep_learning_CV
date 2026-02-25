import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch

from Segmentation_2d.data.dataset import VocSegmentationDataset, CityScapesDataset

DECODE_DICT = {
    "CityScapes": CityScapesDataset,
    "VOC": VocSegmentationDataset,
}

class ImageSegVisualizer:
    def __init__(self, dataset_name, mean, std, alpha=0.6):
        self.dataset = DECODE_DICT[dataset_name]
        self.alpha = alpha
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def denormalize(self, img_tensor):
        img_tensor = img_tensor * self.std + self.mean if self.std is not None else img_tensor + self.mean
        img_tensor = img_tensor.permute(0, 2, 3, 1).numpy()
        return img_tensor

    def visualize(self, masks, input_dict, save_path):
        imgs = self.denormalize(input_dict['img'])
        imgs = (imgs * 255).astype(np.uint8)
        colorized_masks = self.dataset.decode_target(masks).astype('uint8')
        imgs_f = imgs.astype(np.float32)
        colorized_masks_f = colorized_masks.astype(np.float32)
        overlays = ((1 - self.alpha) * imgs_f + self.alpha * colorized_masks_f).astype(np.uint8)
        bs = imgs.shape[0]
        ori_sizes = input_dict['original_size']
        _, axs = plt.subplots(3, bs, figsize=(8, 8))

        for i in range(bs):
            ori_size = ori_sizes[i]
            img = imgs[i]
            colorized_mask = colorized_masks[i]
            overlay = overlays[i]

            if "padding" in input_dict and "rescale_size" in input_dict:
                paddings = input_dict['padding']
                rescale_sizes = input_dict['rescale_size']
                dw, dh = paddings[i]
                nw, nh = rescale_sizes[i]

                img = img[dh:nh+dh, dw:dw+nw]
                colorized_mask = colorized_mask[dh:nh+dh, dw:dw+nw]
                overlay = overlay[dh:nh+dh, dw:dw+nw]

            img = Image.fromarray(img).resize((ori_size[1], ori_size[0]), resample=Image.NEAREST)
            colorized_mask = Image.fromarray(colorized_mask).resize((ori_size[1], ori_size[0]), resample=Image.NEAREST)
            overlay = Image.fromarray(overlay).resize((ori_size[1], ori_size[0]), resample=Image.NEAREST)

            axs[0, i].imshow(colorized_mask)
            axs[1, i].imshow(img)
            axs[2, i].imshow(overlay)
            axs[0, i].axis('off')
            axs[1, i].axis('off')
            axs[2, i].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "seg_result.png"), bbox_inches='tight')
        plt.show()