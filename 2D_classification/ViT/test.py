import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader
import math

from model import *
from dataset import *

def test_model(args):
    device = args.device
    dataset_type = args.dataset
    batch_size = args.batch_size
    path = args.data_path
    model_name = args.model
    
    print("Start test model {} on {} dataset!".format(model_name, dataset_type))
    
    rows = cols = int(math.sqrt(batch_size))

    if dataset_type == "flower":
        data_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        valDataset = flowerDataset(path, data_transform)
        valDataloader = DataLoader(valDataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=valDataset.collate_fn)
        json_path = '../../Dataset/flower_data/classIndex.json'
        with open(json_path, "r") as f:
            class_indict = json.load(f)
        class_num = 5
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    if model_name == "vit_sinusoidal":
        model = ViT_sinusoidal(class_num=class_num).to(device)
    elif model_name == "vit_relative":
        model = ViT_relative(class_num=class_num).to(device)
    elif model_name == "vit_rope":
        model = ViT_rope(class_num=class_num).to(device)
    else:
        raise ValueError(f'unknown model {model_name}')
    
    weights_path = "ckpts/{}_{}.pth".format(model_name, dataset_type)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    plt.figure(figsize=(4 * cols, 4 * rows))
    for imgs, _ in valDataloader:
        imgs_denorm = imgs * std + mean
        with torch.no_grad():
            output = torch.squeeze(model(imgs.to(device))).cpu()
            predict = torch.softmax(output, dim=1)
            predict_cla = torch.argmax(predict, dim=1).numpy()
            
            for i in range(batch_size):
                plt.subplot(rows, cols, i+1)
                plt.imshow(imgs_denorm[i].permute(1, 2, 0).numpy())
                plt.axis('off')
                
                class_name = class_indict[str(predict_cla[i])]
                score = predict[i, predict_cla[i]].numpy()
                title = "{}, score: {:.2f}".format(class_name, score)
                plt.title(title, fontsize=10)
            
            plt.tight_layout()
            plt.show()
            plt.savefig('img/{}_{}.png'.format(model_name, dataset_type), bbox_inches='tight')
            break
    
def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', type=str, default="flower")
    parse.add_argument('--model', type=str, default="vit_relative")
    parse.add_argument('--batch_size', type=int, default=4)
    parse.add_argument('--data_path', type=str, default="../../Dataset/flower_data/val")
    parse.add_argument('--device', type=str, default="cuda")
    args = parse.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    test_model(args)
