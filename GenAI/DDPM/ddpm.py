import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from model import Unet
import logging
import argparse
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Diffusion:
    def __init__(self, noise_steps=1000, beta_s=1e-4, beta_e=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_s = beta_s
        self.beta_e = beta_e
        self.img_size = img_size
        self.device = device
        
        self.beta = self.noiseSchedule().to(device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
    def noiseSchedule(self):
        return torch.linspace(self.beta_s, self.beta_e, self.noise_steps)
    
    def noise_imgs(self, x0, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_completed_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Z = torch.randn_like(x0)
        return sqrt_alpha_hat * x0 + sqrt_completed_alpha_hat * Z, Z
    
    def sample_timesteps(self, n):
        "This function generate a batch of sampling timesteps"
        return torch.randint(low=1, high=self.noise_steps, size=(n, ))
    
    def sample(self, model: nn.Module, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad:
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                pred_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                    
                x = 1 / torch.sqrt(alpha) * (x - ((1-alpha) / (torch.sqrt(1-alpha_hat))) * pred_noise) + torch.sqrt(beta) * noise
        
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
    
def train_model(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = getData(args)
    model = Unet().to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}: ")
        pbar = tqdm(dataloader)
        for i, (imgs, _) in enumerate(pbar):
            imgs = imgs.to(device)
            t = diffusion.sample_timesteps(imgs.shape[0]).to(device)
            xt, noise = diffusion.noise_imgs(imgs, t)
            pred_noise = model(xt, t)
            loss = mse(noise, pred_noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
            
        sampledImgs = diffusion.sample(model, n = imgs.shape[0])
        saveImg(sampledImgs, os.path.join("results", args.run_name, f"{epoch}.png"))
        torch.save(model.state_dict(), os.path.join("ckpts", args.run_name, f"ckpt.pt"))
            

def parse_args():
    parser = argparse.ArgumentParser()
    parse.add_argument('--run_name', type=str, default="DDPM_Unconditional", help='Model Type')
    parse.add_argument('--epochs', type=int, default=500)
    parse.add_argument('--batch_size', type=int, default=12)
    parse.add_argument('--img_size', type=int, default=64)
    parse.add_argument('--data_path', type=str, default="/project/datasets/landscape_img_folder")
    parse.add_argument('--device', type=str, default="cuda")
    parse.add_argument('--lr', type=float, default=3e-4)
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    train_model(args)
    