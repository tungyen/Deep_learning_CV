import torch
from tqdm import tqdm
from dataset import *
from model import *
from utils import *

import os
import argparse

def eval_model(args):
    ckpts_path = "ckpts"
    device = args.device
    model_name = args.model
    dataset_type = args.dataset
    weight_path = os.path.join(ckpts_path, '{}_{}.pth'.format(model_name, dataset_type))
    
    print("Start evaluation model {} on {} dataset!".format(model_name, dataset_type))
    
    _, valDataloader, _, _ = get_dataset(args)
    val_num = len(valDataloader.dataset)
    model = get_model(args)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    
    acc = 0.0
    with torch.no_grad():
        for img, label in tqdm(valDataloader):
            output = model(img.to(device))
            pred_class = torch.argmax(output, dim=1)
            acc += torch.eq(pred_class, label.to(device)).sum().item()
    acc = acc / val_num
    print("Validation Acc===>{}".format(acc))

            
def parse_args():
    parse = argparse.ArgumentParser()
    # Dataset
    parse.add_argument('--dataset', type=str, default="modelnet40")
    parse.add_argument('--n_points', type=int, default=1500)
    
    # Model
    parse.add_argument('--model', type=str, default="pointnet_cls")
    parse.add_argument('--class_num', type=int, default=40)
    
    # Eval
    parse.add_argument('--batch_size', type=int, default=16)
    parse.add_argument('--device', type=str, default="cuda")
    args = parse.parse_args()
    return args
        
if __name__ =='__main__':
    args = parse_args()
    eval_model(args)