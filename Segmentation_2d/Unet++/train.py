import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler

from metrics import ConfusionMatrix
from dataset import *
from model import Unet_plus2
import transforms as T

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', default=1, type=int, help='Total cetegory of dataset')
    parser.add_argument('--batch_size', default=4, type=int, help='Number of batch size')
    parser.add_argument('--deep_supervise', default=False, type=bool)
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--min_lr', default=1e-5, type=float, help='minimum learning rate')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    args = parser.parse_args()
    return args

class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)
    
    
def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 520
    crop_size = 480

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(base_size, mean=mean, std=std)


def Unet_plus2_train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weightPath = 'Unet_plus2.pt'
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 4])
    
    train_dataset = VOCSegmentation(voc_root='../Dataset', transforms=get_transform(train=True))
    val_dataset = VOCSegmentation(voc_root='../Dataset', transforms=get_transform(train=False), txt_name='val.txt')
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=nw, pin_memory=True, collate_fn=train_dataset.collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=nw, pin_memory=True, collate_fn=val_dataset.collate_fn)
    
    model = Unet_plus2(numClass=args.num_class, deep_supervise=args.deep_supervise).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    
    iou_best = 0
    for epoch in range(args.epochs):
        print("Epoch {} starts!".format(epoch))
        
        # train
        model.train()
        for img, mask in tqdm(train_loader):
            img, mask = img.to(device), mask.to(device)
            
            pred = model(img)
            loss = nn.functional.cross_entropy(pred, mask, ignore_index=255)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        print("Epoch {}-training loss===>{}".format(epoch, loss.item()))
        
        # Val
        model.eval()
        confmat = ConfusionMatrix(args.num_class)
        for img, mask in tqdm(val_loader):
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            confmat.update(mask.flatten(), pred.argmax(1).flatten())
            
        acc_global, acc, iou = confmat.compute()
        mIoU = torch.mean(iou).cpu().item()
        print("Epoch {}-Global Accuracy===>{}, Average IoU===>{}.".format(epoch, acc_global.cpu().item(), mIoU))
        if mIoU > iou_best:
            iou_best = mIoU
            torch.save(model.state_dict(), weightPath)
            
    
if __name__ == '__main__':
    args = parse_args()
    Unet_plus2_train(args)