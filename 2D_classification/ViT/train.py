import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import math
import os
import argparse

from dataset import *
from model import *
from utils import *


def train_model(args):
    os.makedirs("ckpts", exist_ok=True)
    device = args.device
    dataset_type = args.dataset
    model_name = args.model
    epochs = args.epochs
    lr = args.lr
    m = args.momentum
    weight_decay = args.weight_decay
    lrf = args.lrf
    
    weight_path = os.path.join("ckpts", "{}_{}.pth".format(model_name, dataset_type))
    print("Start training model {} on {} dataset!".format(model_name, dataset_type))

    trainDataloader, valDataloader, _, _, val_num = get_dataset(args)
    model = get_model(args)
    # model.load_state_dict(torch.load(weight_path, map_location=device))
    # model.eval()
        
    opt = optim.SGD(model.parameters(), lr=lr, momentum=m, weight_decay=weight_decay)
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf
    scheduler = lr_scheduler.LambdaLR(opt, lr_lambda=lf)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(epochs):
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
                predClass = torch.argmax(output, dim=1)
                acc += torch.eq(predClass, label.to(device)).sum().item()
        acc = acc / val_num
        print("Epoch {}-validation Acc===>{}".format(epoch+1, acc))
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), weight_path)
            
def parse_args():
    parse = argparse.ArgumentParser()
    # Dataset
    parse.add_argument('--dataset', type=str, default="cifar10")
    parse.add_argument('--data_path', type=str, default="../../Dataset/flower_data")
    
    # Model
    parse.add_argument('--model', type=str, default="vit_relative")
    parse.add_argument('--img_size', type=int, default=32)
    parse.add_argument('--patch_size', type=int, default=4)
    parse.add_argument('--class_num', type=int, default=10)
    
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