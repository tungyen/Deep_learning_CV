from dataset import *
from model import *
import torch
from tqdm import tqdm
from torch import optim
import os
import argparse

def train_model(args):
    dataset = CelebA_dataset(args.root, img_size=args.img_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    device = args.device

    model = VAE(3, args.latent_dim, img_size=args.img_size).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    img_dim = 3 * args.img_size * args.img_size
    
    def VAE_loss(x, reconstruct, mu, log_var, k_weight=0.00025):
        recons_loss = F.mse_loss(reconstruct, x.view(-1, img_dim))
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recons_loss + k_weight * kl_loss
        return {'loss': loss, 'Reconstruction_loss': recons_loss.detach(), 'KL_loss': kl_loss.detach()}

    model.train()
    for _ in range(args.epochs):
        pbar = tqdm(dataloader)
        for img in pbar:
            img = img.to(device)
            [reconstruct, mu, log_var] = model(img)
            log_var = torch.clamp_(log_var, -10, 10)
            loss = VAE_loss(img, reconstruct, mu, log_var)
            opt.zero_grad()
            loss['loss'].backward()
            opt.step()
            pbar.set_postfix(Loss=loss['loss'].item())
            
        torch.save(model.state_dict(), os.path.join("ckpts", "vae.pt"))

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--epochs', type=int, default=20)
    parse.add_argument('--img_size', type=int, default=64)
    parse.add_argument('--latent_dim', type=int, default=128)
    parse.add_argument('--batch_size', type=int, default=16)
    parse.add_argument('--device', type=str, default="cuda")
    parse.add_argument('--lr', type=float, default=1e-3)
    parse.add_argument('--weight_decay', type=float, default=0.0)
    parse.add_argument('--root', type=str, default="../../../Dataset/img_align_celeba")
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    train_model(args)