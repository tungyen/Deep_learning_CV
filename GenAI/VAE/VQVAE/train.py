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
    prior_only = args.prior_only
    
    model_name = "VQVAE" if not prior_only else "VQVAE_prior"
    print("Start training {} model!".format(model_name))

    model = VQVAE(3, args.emb_num, args.emb_dim, args.n_block, args.prior_dim).to(device)
    if prior_only:
        ckpts = torch.load(args.ckpts_path)
        model.load_state_dict(ckpts)
        
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_func = model.VQVAE_loss if not prior_only else model.prior_loss

    for _ in range(args.epochs):
        pbar = tqdm(dataloader)
        for img in pbar:
            img = img.to(device)
            
            output = model(img, prior_only)
            loss = loss_func(img, output)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(Loss=loss.item())
            
        torch.save(model.state_dict(), os.path.join("ckpts", "{}.pt".format(model_name)))

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--epochs', type=int, default=20)
    parse.add_argument('--emb_num', type=int, default=512)
    parse.add_argument('--emb_dim', type=int, default=64)
    parse.add_argument('--n_block', type=int, default=15)
    parse.add_argument('--prior_dim', type=int, default=256)
    parse.add_argument('--img_size', type=int, default=64)
    parse.add_argument('--batch_size', type=int, default=16)
    parse.add_argument('--device', type=str, default="cuda")
    parse.add_argument('--lr', type=float, default=1e-3)
    parse.add_argument('--weight_decay', type=float, default=0.0)
    parse.add_argument('--root', type=str, default="../../../Dataset/img_align_celeba")
    parse.add_argument('--prior_only', type=bool, default=True)
    parse.add_argument('--ckpts_path', type=str, default="ckpts/VQVAE.pt")
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    train_model(args)