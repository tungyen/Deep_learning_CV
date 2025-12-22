import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import math
import os
import argparse

from Classification_2d.ViT.utils import get_model
from Classification_2d.dataset import get_dataset
from Classification_2d.metrics import compute_image_cls_metrics


def train_model(args):
    root = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(root, "ckpts"), exist_ok=True)
    device = args.device
    dataset_type = args.dataset
    model_name = args.model
    epochs = args.epochs
    lr = args.lr
    m = args.momentum
    weight_decay = args.weight_decay
    lrf = args.lrf
    
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
    
    weight_path = os.path.join(root, "ckpts", "{}_{}.pth".format(model_name, dataset_type))
    print("Start training model {} on {} dataset!".format(model_name, dataset_type))

    train_dataloader, val_dataloader, _, _ = get_dataset(args)
    model = get_model(args)
        
    opt = optim.SGD(model.parameters(), lr=lr, momentum=m, weight_decay=weight_decay)
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf
    scheduler = lr_scheduler.LambdaLR(opt, lr_lambda=lf)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(epochs):
        print("Epoch {} start now!".format(epoch+1))
        # Train
        with tqdm(train_dataloader, desc="Training") as pbar:
            for imgs, labels in pbar:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                opt.step()  
                opt.zero_grad()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            scheduler.step()
        
        # Validation
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for imgs, labels in tqdm(val_dataloader, desc="Evaluation"):
                output = model(imgs.to(device))
                pred_classes = torch.argmax(output, dim=1)
                
                all_preds.append(pred_classes.cpu())
                all_labels.append(labels)
        
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
                
        accuracy, precision, recall = compute_image_cls_metrics(args, all_preds, all_labels)
        print("Epoch {}-validation Accuracy===>{:.4f}".format(epoch+1, accuracy))
        print("Epoch {}-validation Precision===>{:.4f}".format(epoch+1, precision))
        print("Epoch {}-validation Recall===>{:.4f}".format(epoch+1, recall))
        if precision > best_acc:
            best_acc = precision
            torch.save(model.state_dict(), weight_path)
            
def parse_args():
    parse = argparse.ArgumentParser()
    # Dataset
    parse.add_argument('--dataset', type=str, default="cifar10")
    parse.add_argument('--data_path', type=str, default="Dataset/flower_data")
    
    # Model
    parse.add_argument('--model', type=str, default="vit_relative")
    
    # training
    parse.add_argument('--epochs', type=int, default=200)
    parse.add_argument('--batch_size', type=int, default=128)
    parse.add_argument('--device', type=str, default="cuda")
    parse.add_argument('--lr', type=float, default=2e-4)
    parse.add_argument('--lrf', type=float, default=0.01)
    parse.add_argument('--momentum', type=float, default=0.9)
    parse.add_argument('--weight_decay', type=float, default=5e-5)
    args = parse.parse_args()
    return args
        
if __name__ =='__main__':
    args = parse_args()
    train_model(args)