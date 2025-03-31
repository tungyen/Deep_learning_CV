from dataset import *
from model import *
import torch
from tqdm import tqdm
from torch import optim
import os
import argparse
from torchvision import datasets

def train_model(args):
    dataset_type = args.dataset
    B = args.batch_size
    device = args.device
    
    if dataset_type == "celeba":
        dataset = CelebA_dataset("../../../Dataset/img_align_celeba")
        dataloader = DataLoader(dataset, batch_size=B, shuffle=True)
        C = 3
        class_num = None
        img_size = 64
        latent_dim = 128
    elif dataset_type == "MNIST":
        dataloader = DataLoader(datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
                     batch_size=B, shuffle=True, pin_memory=True)
        C = 1
        class_num = 10
        img_size = 28
        latent_dim = 10
    elif dataset_type == 'fashion':
        dataloader = DataLoader(datasets.FashionMNIST('fashion', train=True, download=True, transform=transforms.ToTensor()),
                    batch_size=B, shuffle=True, pin_memory=True)
        C = 1
        class_num = 10
        img_size = 28
        latent_dim = 10
    elif dataset_type == 'cifar':
        dataloader = DataLoader(datasets.CIFAR10('cifar', train=True, download=True, transform=transforms.ToTensor()),
                    batch_size=B, shuffle=True, pin_memory=True)
        C = 3
        class_num = 10
        img_size = 32
        latent_dim = 128
    

    model = VAE(C, latent_dim, class_num, img_size=img_size).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    img_dim = C * img_size * img_size
    
    def VAE_loss(x, reconstruct, mu, log_var, k_weight=args.k_weight):
        recons_loss = F.mse_loss(reconstruct, x.view(-1, img_dim))
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recons_loss + k_weight * kl_loss
        return {'loss': loss, 'Reconstruction_loss': recons_loss.detach(), 'KL_loss': kl_loss.detach()}

    model.train()
    for i in range(args.epochs):
        print("Epoch {}: ".format(i+1))
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
            
            [reconstruct, mu, log_var] = model(img, label)
            log_var = torch.clamp_(log_var, -10, 10)
            loss = VAE_loss(img, reconstruct, mu, log_var)
            opt.zero_grad()
            loss['loss'].backward()
            opt.step()
            pbar.set_postfix(Loss=loss['loss'].item())
            
        torch.save(model.state_dict(), os.path.join("ckpts", "VAE_{}.pt".format(dataset_type)))

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', type=str, default="cifar")
    parse.add_argument('--epochs', type=int, default=20)
    parse.add_argument('--k_weight', type=float, default=0.00025)
    parse.add_argument('--batch_size', type=int, default=16)
    parse.add_argument('--device', type=str, default="cuda")
    parse.add_argument('--lr', type=float, default=1e-3)
    parse.add_argument('--weight_decay', type=float, default=0.0)
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    train_model(args)