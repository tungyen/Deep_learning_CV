from tqdm import tqdm
import argparse
import numpy as np
import os
import torch
import torch.optim as optim

from Object_detection_2d.dataset.utils import get_dataset
from Object_detection_2d.loss import get_loss
from Object_detection_2d.optimizer import get_scheduler
from Object_detection_2d.utils import get_model, setup_args_with_dataset

def train_model(args):
    ckpts_path = args.experiment
    root = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(root, ckpts_path), exist_ok=True)
    model_name = args.model
    dataset_type = args.dataset
    setup_args_with_dataset(dataset_type, args)

    weight_path = os.path.join(root, ckpts_path, "{}_{}.pth".format(model_name, dataset_type))
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
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2), eps=eps)
    scheduler = get_scheduler(args, optimizer)
    criterion = get_loss(args)
    best_metric = 0.0
            
    for epoch in range(epochs):
        # Train
        print("Epoch {} start now!".format(epoch+1))
        model.train()
        batch_losses = []
        
        for imgs, targets in tqdm(train_dataloader):
            
            imgs = imgs.to(device)
            boxes = [t['bboxes'].to(device) for t in targets]
            labels = [t['labels'].to(device) for t in targets]
            pred_boxes, pred_logits = model(imgs)

            loss = criterion(pred_boxes, pred_logits, boxes, labels)
            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        scheduler.step()
        epoch_loss = np.mean(batch_losses)
        print("Epoch {}-training loss===>{:.4f}".format(epoch+1, epoch_loss))

        # Validation

def parse_args():
    parse = argparse.ArgumentParser()
    # Dataset
    parse.add_argument('--dataset', type=str, default="voc")
    parse.add_argument('--crop_size', type=int, default=513)
    parse.add_argument('--voc_data_root', type=str, default="Dataset/VOC")
    parse.add_argument('--voc_year', type=str, default="2012_aug")
    parse.add_argument('--voc_download', type=bool, default=False)
    
    # Model
    parse.add_argument('--model', type=str, default="ssd")
    
    # Training
    parse.add_argument('--experiment', type=str, required=True)
    parse.add_argument('--epochs', type=int, default=200)
    parse.add_argument('--device', type=str, default="cuda")
    parse.add_argument('--scheduler', type=str, default="poly")
    parse.add_argument('--lr', type=float, default=0.01)
    parse.add_argument('--weight_decay', type=float, default=1e-4)
    parse.add_argument('--momentum', type=float, default=0.9)
    parse.add_argument('--step_size', type=int, default=70)
    parse.add_argument('--loss_func', type=str, default="ce")
    args = parse.parse_args()
    return args
     
if __name__ =='__main__':
    args = parse_args()
    train_model(args)
    
    
    
    