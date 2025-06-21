import numpy as np
import matplotlib.pyplot as plt
import os

from Segmentation_2d.dataset.voc import VocDataset
from Segmentation_2d.dataset.cityscapes import CityScapesDataset

def visualize_image_seg(args, masks, imgs, save_path, alpha=0.6):
    batch_size = args.batch_size
    model_name = args.model
    dataset_type = args.dataset
    year = args.voc_year
    
    if dataset_type == "voc":
        colorized_masks = VocDataset.decode_target(masks).astype('uint8')
    elif dataset_type == "cityscapes":
        colorized_masks = CityScapesDataset.decode_target(masks).astype('uint8')
    else:
        raise ValueError(f'Unknown dataset {dataset_type}.')
        
    imgs_f = imgs.astype(np.float32)
    colorized_masks_f = colorized_masks.astype(np.float32)
    overlays = ((1 - alpha) * imgs_f + alpha * colorized_masks_f).astype(np.uint8)
    
    _, axs = plt.subplots(3, batch_size, figsize=(8, 8))
    for i in range(batch_size):
        axs[0, i].imshow(colorized_masks[i])
        axs[1, i].imshow(imgs[i])
        axs[2, i].imshow(overlays[i])
        axs[0, i].axis('off')
        axs[1, i].axis('off')
        axs[2, i].axis('off')

    plt.tight_layout()
    plt.show()
    save_name = '{}_{}'.format(model_name, dataset_type)
    if dataset_type == "voc":
        save_name = save_name + "_{}".format(year)
    plt.savefig(os.path.join(save_path, save_name + ".png"), bbox_inches='tight')