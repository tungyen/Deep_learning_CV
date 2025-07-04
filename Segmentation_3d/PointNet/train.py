import torch
import torch.optim as optim
from tqdm import tqdm
import argparse
import os
import numpy as np

from Segmentation_3d.dataset.utils import get_dataset
from Segmentation_3d.utils import get_model, get_loss, setup_args_with_dataset
from Segmentation_3d.metrics import compute_pcloud_semseg_metrics, compute_pcloud_cls_metrics, compute_pcloud_partseg_metrics

def train_model(args):
    ckpts_path = "ckpts"
    root = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(root, ckpts_path), exist_ok=True)
    model_name = args.model
    task = model_name[-3:]
    dataset_type = args.dataset
    setup_args_with_dataset(dataset_type, args)
    
    task = args.task
    weight_path = os.path.join(root, ckpts_path, "{}_{}_{}.pth".format(model_name, dataset_type, task))
    device = args.device
    lr = args.lr
    beta1= args.beta1
    beta2 = args.beta2
    eps = args.eps
    epochs = args.epochs
    weight_decay = args.weight_decay
    
    print("Start training model {} on {} dataset!".format(model_name, dataset_type))
    train_dataloader, val_dataloader, _, class_dict = get_dataset(args)
    model = get_model(args)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2), eps=eps)
    criterion = get_loss(args)
    best_metric = 0.0
            
    for epoch in range(epochs):
        print("Epoch {} start now!".format(epoch+1))
        for pclouds, *labels in tqdm(train_dataloader):
            pclouds = pclouds.to(device).float()
            
            if len(labels) == 1:
                labels = labels[0].to(device)
                outputs, trans_feats = model(pclouds)
            elif len(labels) == 2:
                cls_labels, labels = labels
                cls_labels = cls_labels.to(device)
                labels = labels.to(device)
                outputs, trans_feats = model(pclouds, cls_labels)
            else:
                raise ValueError(f'Too much input data.')

            loss = criterion(outputs, labels, trans_feats)
            opt.zero_grad()
            loss.backward()
            opt.step()     
        print("Epoch {}-training loss===>{:.4f}".format(epoch+1, loss.item()))
        
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
                    instance2parts, parts2instance = class_dict
                    outputs, _ = model(pclouds.to(device), cls_labels.to(device))
                    outputs = outputs.cpu().numpy()
                    pred_classes = np.zeros((outputs.shape[0], outputs.shape[2])).astype(np.int32)
                    for i in range(outputs.shape[0]):
                        instance = parts2instance[labels[i, 0].item()]
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
            print("Epoch {}-validation Accuracy===>{:.4f}".format(epoch+1, accuracy))
            print("Epoch {}-validation Precision===>{:.4f}".format(epoch+1, precision))
            print("Epoch {}-validation Recall===>{:.4f}".format(epoch+1, recall))

            if precision > best_metric:
                best_metric = precision
                torch.save(model.state_dict(), weight_path)

        elif task == "semseg":
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()
            class_ious, miou = compute_pcloud_semseg_metrics(args, all_preds, all_labels)
            print("Validation mIoU===>{:.4f}".format(miou))
            for cls in class_dict:
                print("{} IoU: {:.4f}".format(class_dict[cls], class_ious[cls]))
            if miou > best_metric:
                best_metric = miou
                torch.save(model.state_dict(), weight_path)

        elif task == 'partseg':
            instance_ious, instance_mious, class_mious = compute_pcloud_partseg_metrics(all_preds, all_labels, class_dict)
            print("Validation instance mIoU===>{:.4f}".format(instance_mious))
            print("Validation class mIoU===>{:.4f}".format(class_mious))
            for instance, miou in instance_ious.items():
                print("{} instance mIoU: {:.4f}".format(instance, miou))
                
            if instance_mious > best_metric:
                best_metric = instance_mious
                torch.save(model.state_dict(), weight_path)
        else:
            raise ValueError(f'Unknown segmentation task {task}.')  

def parse_args():
    parse = argparse.ArgumentParser()
    # Dataset
    parse.add_argument('--dataset', type=str, default="s3dis")
    
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
    
    # training
    parse.add_argument('--epochs', type=int, default=200)
    parse.add_argument('--batch_size', type=int, default=64)
    parse.add_argument('--device', type=str, default="cuda")
    parse.add_argument('--lr', type=float, default=1e-3)
    parse.add_argument('--beta1', type=float, default=0.9)
    parse.add_argument('--beta2', type=float, default=0.999)
    parse.add_argument('--eps', type=float, default=1e-8)
    parse.add_argument('--weight_decay', type=float, default=1e-4)
    parse.add_argument('--loss_func', type=str, default="focal")
    args = parse.parse_args()
    return args
     
if __name__ =='__main__':
    args = parse_args()
    train_model(args)
    
    
    
    