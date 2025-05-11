import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import math
from dataset import *
from model import *

import os
import argparse

def train_model(args):
    ckpts_path = "ckpts"
    if not os.path.exists(ckpts_path):
        os.mkdir(ckpts_path)
        
    device = args.device
    path = args.data_path
    dataset_type = args.dataset
    model_name = args.model
    batchSize = args.batch_size
    numEpoch = args.epochs
    lr = args.lr
    m = args.momentum
    weight_decay = args.weight_decay
    lrf = args.lrf
    
    weightPath = os.path.join(ckpts_path, "{}_{}.pth".format(model_name, dataset_type))
    print("Start training model {} on {} dataset!".format(model_name, dataset_type))
    
    if dataset_type == "flower":
        class_num = 5
        data_transform = {
            "train": transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            "val": transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
        train_path = os.path.join(path, "train")
        val_path = os.path.join(path, "val")
        
        trainDataset = flowerDataset(train_path, data_transform["train"])
        valDataset = flowerDataset(val_path, data_transform["val"])
        nw = min([os.cpu_count(), batchSize if batchSize > 1 else 0, 8])
    
        trainDataloader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, pin_memory=True, num_workers=nw, collate_fn=trainDataset.collate_fn)
        valDataloader = DataLoader(valDataset, batch_size=batchSize, shuffle=False, pin_memory=True, num_workers=nw, collate_fn=valDataset.collate_fn)
    else:
        raise ValueError(f'unknown dataset {dataset_type}')
    
    bestAcc = 0
    valNum = len(valDataset)
    
    if model_name == "vit_sinusoidal":
        model = ViT_sinusoidal(class_num=class_num).to(device)
    elif model_name == "vit_relative":
        model = ViT_relative(class_num=class_num).to(device)
    elif model_name == "vit_rope":
        model = ViT_rope(class_num=class_num).to(device)
    else:
        raise ValueError(f'unknown model {model_name}')
        
        
    opt = optim.SGD(model.parameters(), lr=lr, momentum=m, weight_decay=weight_decay)
    lf = lambda x: ((1 + math.cos(x * math.pi / numEpoch)) / 2) * (1 - lrf) + lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(opt, lr_lambda=lf)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(numEpoch):
        print("Epoch {} start now!".format(epoch+1))
        # train
        for img, label in tqdm(trainDataloader):
            img, label = img.to(device), label.to(device)
            output = model(img)
            trainLoss = criterion(output, label)
            trainLoss.backward()
            opt.step()  
            opt.zero_grad()
        scheduler.step()
        print("Epoch {}-training loss===>{}".format(epoch+1, trainLoss.item()))
        
        # Validation
        acc = 0.0
        with torch.no_grad():
            for img, label in tqdm(valDataloader):
                output = model(img.to(device))
                predClass = torch.max(output, dim=1)[1]
                acc += torch.eq(predClass, label.to(device)).sum().item()
        valAcc = acc / valNum
        print("Epoch {}-validation Acc===>{}".format(epoch+1, valAcc))
        if valAcc > bestAcc:
            bestAcc = valAcc
            torch.save(model.state_dict(), weightPath)
            
def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', type=str, default="flower")
    parse.add_argument('--model', type=str, default="vit_rope")
    parse.add_argument('--data_path', type=str, default="../../Dataset/flower_data")
    parse.add_argument('--epochs', type=int, default=200)
    parse.add_argument('--batch_size', type=int, default=4)
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