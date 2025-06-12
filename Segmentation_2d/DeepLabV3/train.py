import os
import torch
import numpy as np
from tqdm import tqdm
import argparse

from dataset import get_dataset
from utils import get_model, get_criterion, get_scheduler
from metrics import compute_image_seg_metrics


def train_model(args):
    os.makedirs("ckpts", exist_ok=True)
    model_name = args.model
    dataset_type = args.dataset
    class_num = args.class_num
    weight_path = "ckpts/{}_{}.pth".format(model_name, dataset_type)
    
    device = args.device
    lr = args.lr
    epochs = args.epochs
    weight_decay = args.weight_decay
    momentum = args.momentum
    
    model = get_model(args)
    train_dataloader, val_dataloader, _, class_dict = get_dataset(args)
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * lr},
        {'params': model.classifier.parameters(), 'lr': lr},
    ], lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = get_scheduler(args, optimizer)
    criterion = get_criterion(args)
    
    print("Start training model {} on {} dataset!".format(model_name, dataset_type))

    best_metric = 0.0
    for epoch in range(epochs):
        print("Epoch {} start now!".format(epoch+1))
        model.train()
        batch_losses = []
        
        # Train
        for imgs, labels in tqdm(train_dataloader):

            imgs = imgs.to(device)
            labels = labels.type(torch.LongTensor).to(device)
            outputs = model(imgs)

            loss = criterion(outputs, labels)
            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        epoch_loss = np.mean(batch_losses)
        print("Epoch {}-training loss===>{:.4f}".format(epoch, epoch_loss))

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        
        for imgs, labels in tqdm(val_dataloader):
            with torch.no_grad():
                outputs = model(imgs.to(device))
                pred_class = torch.argmax(outputs, dim=1)
                
                all_preds.append(pred_class.cpu())
                all_labels.append(labels)
                
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        class_ious, miou = compute_image_seg_metrics(args, all_preds, all_labels)
        print("Validation mIoU===>{:.4f}".format(miou))
        for cls in range(class_num):
            print("{} IoU: {:.4f}".format(class_dict[cls], class_ious[cls]))
            
        if miou > best_metric:
            best_metric = miou
            torch.save(model.state_dict(), weight_path)
    
    
def parse_args():
    parse = argparse.ArgumentParser()
    # Dataset
    parse.add_argument('--dataset', type=str, default="cityscapes")
    
    # Model
    parse.add_argument('--model', type=str, default="deeplabv3")
    parse.add_argument('--backbone', type=str, default="resnet101")
    parse.add_argument('--class_num', type=int, default=19)
    
    # Training
    parse.add_argument('--epochs', type=int, default=100)
    parse.add_argument('--batch_size', type=int, default=32)
    parse.add_argument('--device', type=str, default="cuda")
    parse.add_argument('--scheduler', type=str, default="poly")
    parse.add_argument('--lr', type=float, default=1e-3)
    parse.add_argument('--weight_decay', type=float, default=1e-4)
    parse.add_argument('--momentum', type=float, default=0.9)
    parse.add_argument('--bn_momentum', type=float, default=0.1)
    parse.add_argument('--step_size', type=int, default=10000)
    parse.add_argument('--total_itrs', type=int, default=30e3)
    args = parse.parse_args()
    return args

if __name__ =='__main__':
    args = parse_args()
    train_model(args)