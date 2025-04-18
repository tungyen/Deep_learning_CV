from model import *
import torch
from tqdm import tqdm
from torch import optim
import os
import argparse
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
import sys

def train_model(args):
    
    os.makedirs("ckpts", exist_ok=True)
    
    device = args.device
    dataset_type = args.dataset
    B = args.batch_size
    latent_dim = args.latent_dim
    b1 = args.b1
    b2 = args.b2
    
    if dataset_type == "SVHN":
        dataloader = DataLoader(datasets.SVHN('data', split='train', download=True, transform=transforms.ToTensor()),
                     batch_size=B, shuffle=True, pin_memory=True)
        class_num = 10
        img_size = 32
        img_shape = np.array([3, img_size, img_size])
    elif dataset_type == "MNIST":
        T = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5), std=(0.5))])
        dataloader = DataLoader(datasets.MNIST('data', train=True, download=True, transform=T),
                     batch_size=B, shuffle=True, pin_memory=True)
        class_num = 10
        img_size = 28
        img_shape = np.array([1, img_size, img_size])
        
    model_name = "Conditional_GAN"
    print("Start training {} model on {} dataset!".format(model_name, dataset_type))
    G = Generator(latent_dim, class_num, img_shape).to(device)
    D = Discriminator(class_num, img_shape).to(device)
    
    G_opt = optim.Adam(G.parameters(), lr=args.g_lr, betas=(b1, b2))
    D_opt = optim.Adam(D.parameters(), lr=args.d_lr, betas=(b1, b2))
    loss_func = nn.BCELoss()
    
    FloatTensor = torch.cuda.FloatTensor if device == "cuda" else torch.FloatTensor
    
    for i in range(args.epochs):
        print("Current Epoch is {}!".format(i+1))
        pbar = tqdm(dataloader)
        for batch in pbar:
            img, label = batch
            img = img.to(device)
            label = label.to(device)
            batch = img.shape[0]
            valid = Variable(FloatTensor(batch, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch, 1).fill_(0.0), requires_grad=False)
            
            D_opt.zero_grad()
            validity_real = D(img, label)
            d_real_loss = loss_func(validity_real, valid)
            
            z = torch.randn(batch, latent_dim).to(device)
            gen_img = G(z, label)
            
            validity_fake = D(gen_img, label)
            d_fake_loss = loss_func(validity_fake, fake)
            d_loss = d_fake_loss + d_real_loss
            d_loss.backward()
            D_opt.step()
            
            G_opt.zero_grad()
            gen_img = G(z, label)
            validity = D(gen_img, label)
            g_loss = loss_func(validity, valid)
            g_loss.backward()
            G_opt.step()
            pbar.set_postfix(G_loss=g_loss.item(), D_loss=d_loss.item())
            
        torch.save(D.state_dict(), os.path.join("ckpts", "{}_D_{}.pt".format(model_name, dataset_type)))
        torch.save(G.state_dict(), os.path.join("ckpts", "{}_G_{}.pt".format(model_name, dataset_type)))

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', type=str, default="MNIST")
    parse.add_argument('--epochs', type=int, default=50)
    parse.add_argument('--batch_size', type=int, default=64)
    parse.add_argument('--device', type=str, default="cuda")
    parse.add_argument('--g_lr', type=float, default=2e-4)
    parse.add_argument('--d_lr', type=float, default=2e-4)
    parse.add_argument('--latent_dim', type=int, default=100)
    parse.add_argument('--b1', type=float, default=0.5)
    parse.add_argument('--b2', type=float, default=0.999)
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    train_model(args)