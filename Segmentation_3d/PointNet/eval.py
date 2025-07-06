import torch
from tqdm import tqdm
import os
import argparse
import numpy as np

from Segmentation_3d.dataset.utils import get_dataset
from Segmentation_3d.utils import get_model, setup_args_with_dataset
from Segmentation_3d.metrics import compute_pcloud_semseg_metrics, compute_pcloud_cls_metrics, compute_pcloud_partseg_metrics

def eval_model(args):
    root = os.path.dirname(os.path.abspath(__file__))
    device = args.device
    model_name = args.model
    dataset_type = args.dataset
    args = setup_args_with_dataset(dataset_type, args)
    task = args.task
    ckpts_path = "ckpts2"
    weight_path = os.path.join(root, ckpts_path, "{}_{}_{}.pth".format(model_name, dataset_type, task))
    
    print("Start evaluation model {} on {} dataset!".format(model_name, dataset_type))
    _, val_dataloader, _, class_dict = get_dataset(args)
    model = get_model(args)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    
    # Validation
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for pclouds, *labels in tqdm(val_dataloader):
            # Semantic Segmentation or Classification
            if len(labels) == 1:
                labels = labels[0]
                outputs, _ = model(pclouds.to(device))
                pred_classes = torch.argmax(outputs, dim=1).cpu()
            # Part Segmentation
            elif len(labels) == 2:
                cls_labels, labels = labels
                instance2parts, _, label2class = class_dict
                outputs, _ = model(pclouds.to(device), cls_labels.to(device))
                outputs = outputs.cpu().numpy()
                pred_classes = np.zeros((outputs.shape[0], outputs.shape[2])).astype(np.int32)
                for i in range(outputs.shape[0]):
                    instance = label2class[cls_labels[i].item()]
                    logits = outputs[i, :, :]
                    pred_classes[i, :] = np.argmax(logits[instance2parts[instance], :], 0) + instance2parts[instance][0]
            else:
                raise ValueError(f'Too much input data.')

            all_preds.append(pred_classes)
            all_labels.append(labels)

    if task == "cls":
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        accuracy, precision, recall = compute_pcloud_cls_metrics(args, all_preds, all_labels)
        print("Validation Accuracy===>{:.4f}".format(accuracy))
        print("Validation Precision===>{:.4f}".format(precision))
        print("Validation Recall===>{:.4f}".format(recall))

    elif task == "semseg":
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        class_ious, miou = compute_pcloud_semseg_metrics(args, all_preds, all_labels)
        print("Validation mIoU===>{:.4f}".format(miou))
        for cls in class_dict:
            print("{} IoU: {:.4f}".format(class_dict[cls], class_ious[cls]))
            
    elif task == 'partseg':
        instance_ious, instance_mious, class_mious = compute_pcloud_partseg_metrics(all_preds, all_labels, class_dict)
        print("Validation instance mIoU===>{:.4f}".format(instance_mious))
        print("Validation class mIoU===>{:.4f}".format(class_mious))
        for instance, miou in instance_ious.items():
            print("{} instance mIoU: {:.4f}".format(instance, miou))
    else:
        raise ValueError(f'Unknown segmentation task {task}.')  

def parse_args():
    parse = argparse.ArgumentParser()
    # Dataset
    parse.add_argument('--dataset', type=str, default="shapenet")
    
    # S3DIS
    parse.add_argument('--test_area', type=int, default=5)
    parse.add_argument('--max_dropout', type=float, default=0.95)
    parse.add_argument('--block_type', type=str, default='static')
    parse.add_argument('--block_size', type=float, default=1.0)
    
    # ShapeNet
    parse.add_argument('--normal_channel', type=bool, default=True)
    parse.add_argument('--class_choice', type=list, default=None)
    
    # Model
    parse.add_argument('--model', type=str, default="pointnet")
    
    # Eval
    parse.add_argument('--device', type=str, default="cuda")
    args = parse.parse_args()
    return args
        
if __name__ =='__main__':
    args = parse_args()
    eval_model(args)