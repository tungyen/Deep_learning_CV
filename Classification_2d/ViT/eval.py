import torch
from tqdm import tqdm
import os
import argparse

from Classification_2d.ViT.utils import get_model
from Classification_2d.dataset import get_dataset
from Classification_2d.metrics import compute_image_cls_metrics

def eval_model(args):
    root = os.path.dirname(os.path.abspath(__file__))
    device = args.device
    model_name = args.model
    dataset_type = args.dataset
    weight_path = os.path.join(root, "ckpts", '{}_{}.pth'.format(model_name, dataset_type))
    
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
    
    print("Start evaluation model {} on {} dataset!".format(model_name, dataset_type))
    
    _, val_dataloader, _, _ = get_dataset(args)
    model = get_model(args)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(val_dataloader, desc="Evaluation"):
            outputs = model(imgs.to(device))
            pred_classes = torch.argmax(outputs, dim=1)
            
            all_preds.append(pred_classes.cpu())
            all_labels.append(labels)
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
            
    accuracy, precision, recall = compute_image_cls_metrics(args, all_preds, all_labels)
    print("Validation Accuracy===>{:.4f}".format(accuracy))
    print("Validation Precision===>{:.4f}".format(precision))
    print("Validation Recall===>{:.4f}".format(recall))

            
def parse_args():
    parse = argparse.ArgumentParser()
    # Dataset
    parse.add_argument('--dataset', type=str, default="flower")
    parse.add_argument('--data_path', type=str, default="Dataset/flower_data")
    
    # Model
    parse.add_argument('--model', type=str, default="vit_rope")
    
    # evaluating
    parse.add_argument('--batch_size', type=int, default=128)
    parse.add_argument('--device', type=str, default="cuda")
    args = parse.parse_args()
    return args
        
if __name__ =='__main__':
    args = parse_args()
    eval_model(args)