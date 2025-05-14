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
    
    _, valDataloader, _, _, val_num = get_dataset(args)
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
    parse.add_argument('--dataset', type=str, default="cifar100")
    parse.add_argument('--data_path', type=str, default="../../Dataset/flower_data")
    
    # Model
    parse.add_argument('--model', type=str, default="vit_rope")
    parse.add_argument('--img_size', type=int, default=32)
    parse.add_argument('--patch_size', type=int, default=4)
    parse.add_argument('--class_num', type=int, default=100)
    
    # evaluating
    parse.add_argument('--batch_size', type=int, default=128)
    parse.add_argument('--device', type=str, default="cuda")
    args = parse.parse_args()
    return args
        
if __name__ =='__main__':
    args = parse_args()
    eval_model(args)