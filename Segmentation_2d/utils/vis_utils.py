import numpy as np
import matplotlib.pyplot as plt
import os

from Segmentation_2d.data.dataset import VocSegmentationDataset, CityScapesDataset

DECODE_DICT = {
    "CityScapes": CityScapesDataset,
    "VOC": VocSegmentationDataset,
}

def visualize_image_seg(dataset_name, bs, masks, imgs, save_path, alpha=0.6):
    colorized_masks = DECODE_DICT[dataset_name].decode_target(masks).astype('uint8')
    imgs_f = imgs.astype(np.float32)
    colorized_masks_f = colorized_masks.astype(np.float32)
    overlays = ((1 - alpha) * imgs_f + alpha * colorized_masks_f).astype(np.uint8)
    
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