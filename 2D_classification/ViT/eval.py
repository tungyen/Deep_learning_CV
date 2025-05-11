import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import *
from model import *

import os
import argparse

def eval_model(args):
    ckpts_path = "ckpts"
    device = args.device
    path = args.data_path
    model_name = args.model
    dataset_type = args.dataset
    weightPath = os.path.join(ckpts_path, '{}_{}.pth'.format(model_name, dataset_type))
    batchSize = args.batch_size
    
    print("Start evaluation model {} on {} dataset!".format(model_name, dataset_type))
    
    
    if dataset_type == "flower":
        class_num = 5
        data_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        val_path = os.path.join(path, "val")
        valDataset = flowerDataset(val_path, data_transform)
        nw = min([os.cpu_count(), batchSize if batchSize > 1 else 0, 8])
        valDataloader = DataLoader(valDataset, batch_size=batchSize, shuffle=False, pin_memory=True, num_workers=nw, collate_fn=valDataset.collate_fn)
    else:
        raise ValueError(f'unknown dataset {dataset_type}')
    
    
    valNum = len(valDataset)
    if model_name == "vit_sinusoidal":
        model = ViT_sinusoidal(class_num=class_num).to(device)
    elif model_name == "vit_relative":
        model = ViT_relative(class_num=class_num).to(device)
    elif model_name == "vit_rope":
        model = ViT_rope(class_num=class_num).to(device)
    else:
        raise ValueError(f'unknown model {model_name}')
    
    model.load_state_dict(torch.load(weightPath, map_location=device))
    model.eval()
    
    acc = 0.0
    with torch.no_grad():
        for img, label in tqdm(valDataloader):
            output = model(img.to(device))
            predClass = torch.max(output, dim=1)[1]
            acc += torch.eq(predClass, label.to(device)).sum().item()
    valAcc = acc / valNum
    print("Validation Acc===>{}".format(valAcc))

            
def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', type=str, default="flower")
    parse.add_argument('--model', type=str, default="vit_rope")
    parse.add_argument('--data_path', type=str, default="../../Dataset/flower_data")
    parse.add_argument('--batch_size', type=int, default=4)
    parse.add_argument('--device', type=str, default="cuda")
    args = parse.parse_args()
    return args
        
if __name__ =='__main__':
    args = parse_args()
    eval_model(args)