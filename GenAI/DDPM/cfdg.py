import os
import copy
import argparse
import numpy as np
import torch
import math
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from model import Conditional_Unet, EMA
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_s=1e-4, beta_e=0.02, img_size=256, device="cuda", beta_scheduler='linear'):
        self.noise_steps = noise_steps
        self.beta_s = beta_s
        self.beta_e = beta_e
        self.img_size = img_size
        self.device = device
        
        self.beta = self.noiseSchedule(beta_scheduler).to(device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
        
    def linear_beta_schedule(self, timesteps):
        scale = 1000 / timesteps
        beta_start = scale * self.beta_s
        beta_end = scale *self.beta_e
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


    def cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def quadratic_beta_schedule(self, timesteps):
        return torch.linspace(self.beta_s**0.5, self.beta_e**0.5, timesteps) ** 2
    
    def sigmoid_beta_schedule(self, timesteps):
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (self.beta_e - self.beta_s) + self.beta_s
        
        
    def noiseSchedule(self, beta_scheduler):
        if beta_scheduler == 'linear':
            return self.linear_beta_schedule(self.noise_steps)
        elif beta_scheduler == 'cosine':
            return self.cosine_beta_schedule(self.noise_steps)
        elif beta_scheduler == 'quadratic':
            return self.quadratic_beta_schedule(self.noise_steps)
        elif beta_scheduler == 'sigmoid':
            return self.sigmoid_beta_schedule(self.noise_steps)
        else:
            raise ValueError(f'unknown beta schedule {beta_scheduler}')
    
    
    def noise_imgs(self, x0, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_completed_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Z = torch.randn_like(x0)
        return sqrt_alpha_hat * x0 + sqrt_completed_alpha_hat * Z, Z
    
    
    def sample_timesteps(self, n):
        "This function generate a batch of sampling timesteps"
        return torch.randint(low=1, high=self.noise_steps, size=(n, ))
    
    
    def sample(self, model: nn.Module, n, labels, scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                x = x.float()
                pred_noise = model(x, t)
                if scale > 0:
                    unconditional_pred_noise = model(x, t, None)
                    pred_noise = torch.lerp(unconditional_pred_noise, pred_noise, scale)
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
    model = Conditional_Unet().to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}: ")
        pbar = tqdm(dataloader)
        for i, (imgs, labels) in enumerate(pbar):
            imgs = imgs.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(imgs.shape[0]).to(device)
            xt, noise = diffusion.noise_imgs(imgs, t)
            xt = xt.float()
            if np.random.random() < 0.1:
                labels = None
            pred_noise = model(xt, t, labels)
            loss = mse(noise, pred_noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            ema.EMA_step(ema_model, model)
            
            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
            
        if epoch % 10 == 0:
            labels = torch.arange(10).long().to(device)
            sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
            ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
            saveImg(sampled_images, os.path.join("results", args.run_name, f"{epoch}.png"))
            saveImg(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.png"))
            torch.save(model.state_dict(), os.path.join("ckpts", args.run_name, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("ckpts", args.run_name, f"ema_ckpt.pt"))
            

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--run_name', type=str, default="DDPM_conditional", help='Model Type')
    parse.add_argument('--epochs', type=int, default=300)
    parse.add_argument('--batch_size', type=int, default=14)
    parse.add_argument('--img_size', type=int, default=64)
    parse.add_argument('--num_classes', type=int, default=10)
    parse.add_argument('--data_path', type=str, default="../../Dataset/cifar10/cifar10-64/train")
    parse.add_argument('--device', type=str, default="cuda")
    parse.add_argument('--lr', type=float, default=3e-4)
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    train_model(args)