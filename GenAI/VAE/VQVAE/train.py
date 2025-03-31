from dataset import *
from model import *
import torch
from tqdm import tqdm
from torch import optim
import os
import argparse
from torchvision import datasets

def train_model(args):
    device = args.device
    prior_only = args.prior_only
    dataset_type = args.dataset
    B = args.batch_size
    beta = args.beta
    
    if dataset_type == "celeba":
        dataset = CelebA_dataset("../../../Dataset/img_align_celeba")
        dataloader = DataLoader(dataset, batch_size=B, shuffle=True)
        C = 3
        class_num = None
    elif dataset_type == "MNIST":
        dataloader = DataLoader(datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
                     batch_size=B, shuffle=True, pin_memory=True)
        C = 1
        class_num = 10
    elif dataset_type == 'fashion':
        dataloader = DataLoader(datasets.FashionMNIST('fashion', train=True, download=True, transform=transforms.ToTensor()),
                    batch_size=B, shuffle=True, pin_memory=True)
        C = 1
        class_num = 10
    elif dataset_type == 'cifar':
        dataloader = DataLoader(datasets.CIFAR10('cifar', train=True, download=True, transform=transforms.ToTensor()),
                    batch_size=B, shuffle=True, pin_memory=True)
        C = 3
        class_num = 10
        
    model_name = "VQVAE" if not prior_only else "VQVAE_prior"
    print("Start training {} model on {} dataset!".format(model_name, dataset_type))
    model = VQVAE(C, args.emb_num, args.emb_dim, args.n_block, class_num, args.prior_dim, beta=beta).to(device)
    if prior_only:
        ckpts = torch.load("ckpts/VQVAE_{}.pt".format(dataset_type))
        model.load_state_dict(ckpts)

    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_func = model.VQVAE_loss if not prior_only else model.prior_loss

    for i in range(args.epochs):
        print("Current Epoch is {}!".format(i+1))
        pbar = tqdm(dataloader)
        for batch in pbar:
            if dataset_type == "celeba":
                img = batch.to(device)
                label = None
            elif dataset_type == "MNIST":
                img, label = batch
                img = img.to(device)
                label = label.to(device)
            elif dataset_type == 'fashion':
                img, label = batch
                img = img.to(device)
                label = label.to(device)
            elif dataset_type == 'cifar':
                img, label = batch
                img = img.to(device)
                label = label.to(device)
                
            output = model(img, label, prior_only)
            loss = loss_func(img, output)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(Loss=loss.item())
            
        torch.save(model.state_dict(), os.path.join("ckpts", "{}_{}.pt".format(model_name, dataset_type)))

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', type=str, default="fashion")
    parse.add_argument('--epochs', type=int, default=20)
    parse.add_argument('--emb_num', type=int, default=512)
    parse.add_argument('--emb_dim', type=int, default=64)
    parse.add_argument('--n_block', type=int, default=15)
    parse.add_argument('--prior_dim', type=int, default=256)
    parse.add_argument('--beta', type=float, default=1.0)
    parse.add_argument('--batch_size', type=int, default=128)
    parse.add_argument('--device', type=str, default="cuda")
    parse.add_argument('--lr', type=float, default=2e-4)
    parse.add_argument('--weight_decay', type=float, default=0.0)
    parse.add_argument('--prior_only', type=bool, default=True)
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    train_model(args)