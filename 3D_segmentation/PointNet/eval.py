import torch
from tqdm import tqdm
import os
import argparse

from dataset import *
from model import *
from utils import *
from metrics import *


def eval_model(args):
    ckpts_path = "ckpts"
    device = args.device
    model_name = args.model
    dataset_type = args.dataset
    class_num = args.class_num
    weight_path = os.path.join(ckpts_path, '{}_{}.pth'.format(model_name, dataset_type))
    task = model_name[-3:]
    
    print("Start evaluation model {} on {} dataset!".format(model_name, dataset_type))
    
    _, val_dataloader, _, class_dict = get_dataset(args)
    model = get_model(args)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    
    if task == "cls":
        acc = compute_pcloud_classification_metrics(args, val_dataloader, model)
        print("Validation Acc===>{}".format(acc))
        
    elif task == "seg":
        class_ious, miou = compute_pcloud_seg_metrics(args, val_dataloader, model)
        print("Validation mIoU===>{:.4f}".format(miou))
        for cls in range(class_num):
            print("{} IoU: {:.4f}".format(class_dict[cls], class_ious[cls]))

            
def parse_args():
    parse = argparse.ArgumentParser()
    # Dataset
    parse.add_argument('--dataset', type=str, default="chair")
    parse.add_argument('--n_points', type=int, default=1500)
    
    # Model
    parse.add_argument('--model', type=str, default="pointnet_seg")
    parse.add_argument('--class_num', type=int, default=4)
    
    # Eval
    parse.add_argument('--batch_size', type=int, default=16)
    parse.add_argument('--device', type=str, default="cuda")
    args = parse.parse_args()
    return args
        
if __name__ =='__main__':
    args = parse_args()
    eval_model(args)