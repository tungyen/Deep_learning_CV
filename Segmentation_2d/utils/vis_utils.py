import numpy as np
import matplotlib.pyplot as plt
import os

from Segmentation_2d.data.dataset import VocSegmentationDataset, CityScapesDataset

DECODE_DICT = {
    "CityScapes": CityScapesDataset,
    "VOC": VocSegmentationDataset,
}

class ImageSegVisualizer:
    def __init__(self, dataset_name, alpha=0.6):
        self.dataset = DECODE_DICT[dataset_name]
        self.alpha = alpha

    def visualize(masks, imgs, save_path):
        colorized_masks = self.dataset.decode_target(masks).astype('uint8')
        imgs_f = imgs.astype(np.float32)
        colorized_masks_f = colorized_masks.astype(np.float32)
        overlays = ((1 - self.alpha) * imgs_f + self.alpha * colorized_masks_f).astype(np.uint8)
        bs = imgs.shape[0]

        _, axs = plt.subplots(3, bs, figsize=(8, 8))
        for i in range(bs):
            axs[0, i].imshow(colorized_masks[i])
            axs[1, i].imshow(imgs[i])
            axs[2, i].imshow(overlays[i])
            axs[0, i].axis('off')
            axs[1, i].axis('off')
            axs[2, i].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "seg_result.png"), bbox_inches='tight')
        plt.show()