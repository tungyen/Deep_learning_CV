from dataset import *
from model import *
import torch
from tqdm import tqdm
from torch import optim
import os
import argparse

def train_model(args):
    dataset = CelebA_dataset(args.root)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    device = args.device

    model = VQVAE(3, args.emb_num, args.emb_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    img_dim = 3 * args.img_size * args.img_size
    
    def VQVAE_loss(input, reconstruct, VQ_loss):
        input = input.view(-1, img_dim)
        recons_loss = F.mse_loss(reconstruct, input)
        loss = recons_loss + VQ_loss
        return {'loss': loss, 'Reconstruction_loss': recons_loss, 'VQ_loss': VQ_loss}

    for _ in range(args.epochs):
        pbar = tqdm(dataloader)
        for img in pbar:
            img = img.to(device)
            [reconstruct, VQ_loss] = model(img)
            loss = VQVAE_loss(img, reconstruct, VQ_loss)
            opt.zero_grad()
            loss['loss'].backward()
            opt.step()
            pbar.set_postfix(Loss=loss['loss'].item())
            
        torch.save(model.state_dict(), os.path.join("ckpts", "vqvae2.pt"))

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--epochs', type=int, default=20)
    parse.add_argument('--emb_num', type=int, default=512)
    parse.add_argument('--emb_dim', type=int, default=64)
    parse.add_argument('--img_size', type=int, default=64)
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