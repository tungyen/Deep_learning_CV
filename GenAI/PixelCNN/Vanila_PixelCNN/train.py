import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from torch import optim
import os
import argparse
import numpy as np

from model import *

def discretize_imgs(img_tensor, nlevels):
    xnp=img_tensor.numpy()
    xnp_dig=(np.digitize(xnp, np.arange(nlevels) / nlevels) - 1).astype(np.longlong)
    xnp=xnp_dig/(nlevels -1)
    return torch.from_numpy(xnp).float(), torch.from_numpy(xnp_dig)

def train_model(args):
    if args.datasets == 'MNIST':
        dataloader = DataLoader(datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
                     batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
        C = 1
    elif args.datasets == 'CIFAR':
        dataset = datasets.CIFAR10(train=True, download=True, transform=transforms.ToTensor())
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        C = 3
        
    device = args.device
    model = PixelCNN(C, args.n_block, args.h_dim).to(device)
    opt = optim.RMSprop(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(args.epochs):
        pbar = tqdm(dataloader)
        for img, _ in pbar:
            img, img_quant= discretize_imgs(img, args.color_level)
            img = img.to(device)
            img_quant = img_quant.to(device)
            pred = model(img).contiguous()
            loss = criterion(pred, img_quant)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(Cross_Entropy_loss=loss.item())
            
        torch.save(model.state_dict(), os.path.join("ckpts", "pixelCnn_{}_{}.pt".format(args.datasets, args.color_level)))
    

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--datasets', type=str, default='MNIST')
    parse.add_argument('--epochs', type=int, default=25)
    parse.add_argument('--batch_size', type=int, default=16)
    parse.add_argument('--device', type=str, default="cuda")
    parse.add_argument('--n_block', type=int, default=8)
    parse.add_argument('--h_dim', type=int, default=64)
    parse.add_argument('--color_level', type=int, default=256)
    parse.add_argument('--lr', type=float, default=1e-3)
    parse.add_argument('--weight_decay', type=float, default=0.0)
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    train_model(args)