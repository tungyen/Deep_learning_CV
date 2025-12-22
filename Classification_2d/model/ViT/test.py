import os
import torch
import matplotlib.pyplot as plt
import argparse
import math

from Classification_2d.ViT.utils import get_model
from Classification_2d.dataset import get_dataset, get_dataset_stat


def test_model(args):
    root = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(root, "imgs")
    os.makedirs(save_path, exist_ok=True)
    device = args.device
    dataset_type = args.dataset
    batch_size = args.batch_size
    model_name = args.model
    weight_path = os.path.join(root, "ckpts", '{}_{}.pth'.format(model_name, dataset_type))
    print("Start test model {} on {} dataset!".format(model_name, dataset_type))
    
    if dataset_type == "flower":
        args.patch_size = 16
        args.img_size = 224
        args.class_num = 5
    elif dataset_type == "cifar10":
        args.patch_size = 4
        args.img_size = 32
        args.class_num = 10
    elif dataset_type == "cifar100":
        args.patch_size = 4
        args.img_size = 32
        args.class_num = 100
    else:
        raise ValueError(f'Unknown dataset {dataset_type}.')
    
    rows = cols = int(math.sqrt(batch_size))
    mean, std = get_dataset_stat(args)
    _, _, test_dataloader, class_dict = get_dataset(args)
    model = get_model(args)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    
    plt.figure(figsize=(4, 4))
    imgs, _ = next(iter(test_dataloader))
    imgs_denorm = imgs * std + mean
    with torch.no_grad():
        outputs = torch.squeeze(model(imgs.to(device))).cpu()
        predicts = torch.softmax(outputs, dim=1)
        predict_classes = torch.argmax(predicts, dim=1).numpy()
        
        for i in range(batch_size):
            plt.subplot(rows, cols, i+1)
            plt.imshow(imgs_denorm[i].permute(1, 2, 0).numpy())
            plt.axis('off')
            
            class_name = class_dict[str(predict_classes[i])]
            score = predicts[i, predict_classes[i]].numpy()
            title = "{}, score: {:.2f}".format(class_name, score)
            plt.title(title, fontsize=10)
        
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(save_path, '{}_{}.png'.format(model_name, dataset_type)), bbox_inches='tight')
    
def parse_args():
    parse = argparse.ArgumentParser()
    # Dataset
    parse.add_argument('--dataset', type=str, default="cifar10")
    parse.add_argument('--data_path', type=str, default="Dataset/flower_data")
    
    # Model
    parse.add_argument('--model', type=str, default="vit_relative")
    
    # evaluating
    parse.add_argument('--batch_size', type=int, default=4)
    parse.add_argument('--device', type=str, default="cuda")
    args = parse.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    test_model(args)
