import os
import sys
import copy
import argparse
import numpy as np
import torch
import math
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import logging
from torch.utils.tensorboard import SummaryWriter
from diffusers.optimization import get_cosine_schedule_with_warmup
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

sys.path.append("..")
from utils import *
from unet import SR3_UNet, EMA
from dataset import *


class SR3:
    def __init__(self, train_steps=2000, sample_steps=100, beta_s=1e-4, train_beta_e=0.005, sample_beta_e=0.1, img_size=256, device="cuda", beta_scheduler='linear'):
        self.train_steps = train_steps
        self.sample_steps = sample_steps
        self.beta_s = beta_s
        self.train_beta_e = train_beta_e
        self.sample_beta_e = sample_beta_e
        self.img_size = img_size
        self.device = device
        
        self.train_beta = self.noiseSchedule(beta_scheduler, train_steps, train_beta_e).to(device)
        self.sample_beta = self.noiseSchedule(beta_scheduler, sample_steps, sample_beta_e).to(device)
        self.train_alpha = 1 - self.train_beta
        self.sample_alpha = 1 - self.sample_beta
        self.train_alpha_hat = torch.cumprod(self.train_alpha, dim=0)
        self.sample_alpha_hat = torch.cumprod(self.sample_alpha, dim=0)
        self.train_timeSteps = torch.linspace(train_steps-1, 0, sample_steps).long()
        self.sample_timeSteps = torch.linspace(sample_steps-1, 0, sample_steps).long()
        
        
    def linear_beta_schedule(self, timesteps, beta_e):
        scale = 1000 / timesteps
        beta_start = scale * self.beta_s
        beta_end = scale * beta_e
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


    def cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def quadratic_beta_schedule(self, timesteps, beta_e):
        return torch.linspace(self.beta_s**0.5, beta_e**0.5, timesteps) ** 2
    
    def sigmoid_beta_schedule(self, timesteps, beta_e):
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (beta_e - self.beta_s) + self.beta_s
        
        
    def noiseSchedule(self, beta_scheduler, steps, beta_e):
        if beta_scheduler == 'linear':
            return self.linear_beta_schedule(steps, beta_e)
        elif beta_scheduler == 'cosine':
            return self.cosine_beta_schedule(steps)
        elif beta_scheduler == 'quadratic':
            return self.quadratic_beta_schedule(steps, beta_e)
        elif beta_scheduler == 'sigmoid':
            return self.sigmoid_beta_schedule(steps, beta_e)
        else:
            raise ValueError(f'unknown beta schedule {beta_scheduler}')
    
    
    def noise_imgs(self, x0, t):
        gamma_high = self.train_alpha_hat[t-1][:, None, None, None]
        gamma_low = self.train_alpha_hat[t][:, None, None, None]
        gamma = (gamma_high - gamma_low) * torch.rand_like(gamma_high, device=self.device) + gamma_low
        gamma = gamma.to(self.device)
        # sqrt_alpha_hat = torch.sqrt(self.train_alpha_hat[t])[:, None, None, None]
        sqrt_alpha_hat = torch.sqrt(gamma)
        # sqrt_completed_alpha_hat = torch.sqrt(1 - self.train_alpha_hat[t])[:, None, None, None]
        sqrt_completed_alpha_hat = torch.sqrt(1-gamma)
        Z = torch.randn_like(x0)
        return sqrt_alpha_hat * x0 + sqrt_completed_alpha_hat * Z, Z, gamma.squeeze()
    
    def getGammas(self, t):
        gamma_high = self.train_alpha_hat[t-1]
        gamma_low = self.train_alpha_hat[t]
        gamma = (gamma_high - gamma_low) * torch.rand_like(gamma_high, device=self.device) + gamma_low
        gamma = gamma.to(self.device)
        return gamma
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.train_steps, size=(n, ))
    
    
    def sample(self, model: nn.Module, n, imgs_lr):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        imgs_lr = norm(imgs_lr)
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.sample_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                # gamma = self.sample_alpha_hat[t][:, None, None, None]
                gamma = self.getGammas(t).float()
                x = x.float()
                x_concat = torch.cat((x, imgs_lr), dim=1)
                pred_noise = model(x_concat, torch.sqrt(gamma))
                alpha = self.sample_alpha[t][:, None, None, None]
                alpha_hat = self.sample_alpha_hat[t][:, None, None, None]
                beta = self.sample_beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                    
                x = 1 / torch.sqrt(alpha) * (x - ((1-alpha) / (torch.sqrt(1-alpha_hat))) * pred_noise) + torch.sqrt(beta) * noise
        
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
    
    def sample_skip_step(self, model: nn.Module, n, imgs_lr, eta: float = 0.0):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        imgs_lr = norm(imgs_lr)
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for t, s in tqdm(list(zip(self.sample_timeSteps[:-1], self.sample_timeSteps[1:])), desc='Sampling'):
                t = (torch.ones(n) * t).long().to(self.device)
                s = (torch.ones(n) * s).long().to(self.device)
                x = x.float()
                gamma = self.sample_alpha_hat[t][:, None, None, None]
                x_concat = torch.cat((x, imgs_lr), dim=1)
                pred_noise = model(x_concat, torch.sqrt(gamma))

                alpha_t = self.sample_alpha[t][:, None, None, None]
                alpha_hat_t = self.sample_alpha_hat[t][:, None, None, None]
                alpha_hat_s = self.sample_alpha_hat[s][:, None, None, None]
                
                if not math.isclose(eta, 0.0):
                    sigma_t = eta * ((1.0-alpha_hat_s) * (1.0-alpha_t) / (1.0-alpha_hat_t)) ** 0.5
                else:
                    sigma_t = torch.zeros_like(alpha_t)
                    
                # Compute x_0 and first term
                x_0 = (x - ((1.0-alpha_hat_t) ** 0.5) * pred_noise) / ((alpha_hat_t) ** 0.5)
                first_term = ((alpha_hat_s) ** 0.5) * x_0
                
                # Compute second term
                coff = (1.0 - alpha_hat_s-sigma_t**2) ** 0.5
                second_term = coff * pred_noise
                eps = torch.randn_like(x)
                x = first_term + second_term + sigma_t * eps
        
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
    
def train_model(args):
    setup_logging(args.run_name)
    device = args.device
    # dataloader = getData(args)
    dataset = DF2KDataset(args.div_root, args.flickr_root)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    upsample = nn.Upsample(scale_factor=4.0, mode='bicubic').to(device)
    
    model = SR3_UNet().to(device)
    ckpts = torch.load("ckpts/SR3/ema_ckpt.pt")
    model.load_state_dict(ckpts)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(dataloader) * args.epochs),
    )
    mse = nn.MSELoss()
    diffusion = SR3(img_size=args.img_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}: ")
        pbar = tqdm(dataloader)
        for i, (imgs_lr, imgs_hr) in enumerate(pbar):
            imgs_lr = imgs_lr.to(device)
            imgs_lr = norm(torch.clip(upsample(imgs_lr), 0.0, 1.0))
            imgs_hr = norm(imgs_hr.to(device))
            t = diffusion.sample_timesteps(imgs_lr.shape[0]).to(device)
            xt, noise, gamma = diffusion.noise_imgs(imgs_hr, t)
            xt = xt.float()
            gamma = gamma.to(device).float()
            
            x = torch.cat((xt, imgs_lr), dim=1)
            pred_noise = model(x, torch.sqrt(gamma))
            loss = mse(noise, pred_noise)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            opt.step()
            lr_scheduler.step()
            
            ema.EMA_step(ema_model, model)
            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
            
        if epoch % 10 == 0:
            sampled_images = diffusion.sample(model, n=imgs_lr.shape[0], imgs_lr=imgs_lr)
            ema_sampled_images = diffusion.sample(ema_model, n=imgs_lr.shape[0], imgs_lr=imgs_lr)
            saveImg(sampled_images, os.path.join("results", args.run_name, f"{epoch}.png"))
            saveImg(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.png"))
            torch.save(model.state_dict(), os.path.join("ckpts", args.run_name, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("ckpts", args.run_name, f"ema_ckpt.pt"))
            

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', type=str, default="cifar-10", help='Dataset Type')
    parse.add_argument('--run_name', type=str, default="SR3", help='Model Type')
    parse.add_argument('--epochs', type=int, default=500)
    parse.add_argument('--batch_size', type=int, default=4)
    parse.add_argument('--img_size', type=int, default=256)
    parse.add_argument('--div_root', type=str, default="../../../Dataset/DIV2K")
    parse.add_argument('--flickr_root', type=str, default="../../../Dataset/Flickr2K")
    parse.add_argument('--device', type=str, default="cuda")
    parse.add_argument('--lr', type=float, default=1e-4)
    parse.add_argument('--lr_warmup_steps', type=int, default=500)
    parse.add_argument('--train_steps', type=int, default=2000)
    parse.add_argument('--sample_steps', type=int, default=100)
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    train_model(args)