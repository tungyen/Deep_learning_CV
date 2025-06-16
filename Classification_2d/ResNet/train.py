import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_dir)

from utils import get_model
from Classification_2d.dataset import get_dataset
from Classification_2d.metrics import compute_image_cls_metrics

def train_model(args):
    os.makedirs("ckpts", exist_ok=True)
    device = args.device
    dataset_type = args.dataset
    model_name = args.model
    epochs = args.epochs
    lr = args.lr
    weight_decay = args.weight_decay
    
    if dataset_type == "flower":
        args.img_size = 224
        args.class_num = 5
    elif dataset_type == "cifar10":
        args.img_size = 32
        args.class_num = 10
    elif dataset_type == "cifar100":
        args.img_size = 32
        args.class_num = 100
    else:
        raise ValueError(f'Unknown dataset {dataset_type}.')
    
    weight_path = os.path.join("ckpts", "{}_{}.pth".format(model_name, dataset_type))
    print("Start training model {} on {} dataset!".format(model_name, dataset_type))

    train_dataloader, val_dataloader, _, _ = get_dataset(args)
    model = get_model(args)
    # model.load_state_dict(torch.load(weight_path, map_location=device))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc = 0.0
    for epoch in range(epochs):
        
        # Train
        model.train()
        for imgs, labels in tqdm(train_dataloader):
            optimizer.zero_grad()
            logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()

        print("Epoch {}-training loss===>{:.4f}".format(epoch+1, loss.item()))

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for imgs, labels in tqdm(val_dataloader):
                outputs = model(imgs.to(device))
                pred_classes = torch.argmax(outputs, dim=1)
                
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
    parse.add_argument('--dataset', type=str, default="flower")
    parse.add_argument('--data_path', type=str, default="../../Dataset/flower_data")
    
    # Model
    parse.add_argument('--model', type=str, default="resnet34")
    
    # training
    parse.add_argument('--epochs', type=int, default=200)
    parse.add_argument('--batch_size', type=int, default=128)
    parse.add_argument('--device', type=str, default="cuda")
    parse.add_argument('--lr', type=float, default=2e-4)
    parse.add_argument('--weight_decay', type=float, default=5e-5)
    args = parse.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    train_model(args)
