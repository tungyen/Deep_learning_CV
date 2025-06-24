import torch
from tqdm import tqdm
import os
import argparse

from Segmentation_3d.dataset import get_dataset
from Segmentation_3d.utils import get_model
from Segmentation_3d.metrics import compute_pcloud_seg_metrics, compute_pcloud_cls_metrics


def eval_model(args):
    root = os.path.dirname(os.path.abspath(__file__))
    device = args.device
    model_name = args.model
    dataset_type = args.dataset
    
    if dataset_type == 'chair':
        args.class_num = 4
    elif dataset_type == 'modelnet40':
        args.class_num = 40
    else:
        raise ValueError(f'Unknown dataset {dataset_type}.')
    
    class_num = args.class_num
    weight_path = os.path.join(root, "ckpts", '{}_{}.pth'.format(model_name, dataset_type))
    task = model_name[-3:]
    
    print("Start evaluation model {} on {} dataset!".format(model_name, dataset_type))
    
    _, val_dataloader, _, class_dict = get_dataset(args)
    model = get_model(args)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for pclouds, labels in tqdm(val_dataloader):
            outputs = model(pclouds.to(device))
            pred_classes = torch.argmax(outputs, dim=1)
            
            all_preds.append(pred_classes.cpu())
            all_labels.append(labels)
        
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

    if task == "cls":   
        accuracy, precision, recall = compute_pcloud_cls_metrics(args, all_preds, all_labels)
        print("Validation Accuracy===>{:.4f}".format(accuracy))
        print("Validation Precision===>{:.4f}".format(precision))
        print("Validation Recall===>{:.4f}".format(recall))
        
    elif task == "seg":    
        class_ious, miou = compute_pcloud_seg_metrics(args, all_preds, all_labels)
        print("Validation mIoU===>{:.4f}".format(miou))
        for cls in range(class_num):
            print("{} IoU: {:.4f}".format(class_dict[cls], class_ious[cls]))

            
def parse_args():
    parse = argparse.ArgumentParser()
    # Dataset
    parse.add_argument('--dataset', type=str, default="modelnet40")
    parse.add_argument('--n_points', type=int, default=1500)
    
    # Model
    parse.add_argument('--model', type=str, default="pointnet_cls")
    
    # Eval
    parse.add_argument('--batch_size', type=int, default=16)
    parse.add_argument('--device', type=str, default="cuda")
    args = parse.parse_args()
    return args
        
if __name__ =='__main__':
    args = parse_args()
    eval_model(args)