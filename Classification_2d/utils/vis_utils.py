import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import math

class ImageClsVisualizer:
    def __init__(self, class_dict, mean, std):
        self.class_dict = class_dict
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def denormalize(self, img_tensor):
        img_tensor = img_tensor * self.std + self.mean if self.std is not None else img_tensor + self.mean
        img_tensor = img_tensor.permute(0, 2, 3, 1).numpy()
        return img_tensor

    def visualize(self, imgs, labels, scores, save_path):
        imgs = self.denormalize(imgs)
        imgs = (imgs * 255).astype(np.uint8)
        imgs_f = imgs.astype(np.float32)
        bs = imgs.shape[0]
        rows = cols = int(math.sqrt(bs))

        plt.figure(figsize=(4, 4))
        for i in range(bs):
            plt.subplot(rows, cols, i+1)
            plt.imshow(imgs[i])
            plt.axis('off')
            
            class_name = self.class_dict[str(labels[i])]
            score = scores[i, labels[i]]
            title = "{}, score: {:.2f}".format(class_name, score)
            plt.title(title, fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "cls_result.png"), bbox_inches='tight')
        plt.show()