import os
import torch
import numpy as np
from tqdm import tqdm
import argparse

from dataset.utils import get_dataset
from utils import get_model
from metrics import compute_image_seg_metrics


def train_model(args):
    model_name = args.model
    dataset_type = args.dataset
    class_num = args.class_num
    weight_path = "ckpts/{}_{}.pth".format(model_name, dataset_type)
    device = args.device
    
    _, val_dataloader, _, class_dict = get_dataset(args)
    model = get_model(args)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    
    print("Start evaluation model {} on {} dataset!".format(model_name, dataset_type))
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(val_dataloader):
            output = model(imgs.to(device))
            pred_class = torch.argmax(output, dim=1)
            
            all_preds.append(pred_class.cpu())
            all_labels.append(labels)
        
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    class_ious, miou = compute_image_seg_metrics(args, all_preds, all_labels)
    print("Validation mIoU===>{:.4f}".format(miou))
    for cls in range(class_num):
        print("{} IoU: {:.4f}".format(class_dict[cls], class_ious[cls]))
    
def parse_args():
    parse = argparse.ArgumentParser()
    # Dataset
    parse.add_argument('--dataset', type=str, default="cityscapes")
    
    # Model
    parse.add_argument('--model', type=str, default="deeplabv3")
    parse.add_argument('--class_num', type=int, default=19)
    
    # Validation
    parse.add_argument('--batch_size', type=int, default=32)
    parse.add_argument('--device', type=str, default="cuda")
    args = parse.parse_args()
    return args

if __name__ =='__main__':
    args = parse_args()
    train_model(args)