import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from model import Variance_Unet
import logging
import argparse
import math
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from diffusers.optimization import get_cosine_schedule_with_warmup

import numpy as np
from typing import List
from torch.distributed.nn import dist

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class IDDPM:
    def __init__(self, noise_steps=1000, beta_s=1e-4, beta_e=0.02, img_size=256, device="cuda", beta_scheduler='cosine'):
        self.noise_steps = noise_steps
        self.beta_s = beta_s
        self.beta_e = beta_e
        self.img_size = img_size
        self.device = device
        
        self.beta = self.noiseSchedule(beta_scheduler).to(device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.alpha_hat_prev = torch.concat((torch.ones(1).to(self.alpha_hat), self.alpha_hat[:-1])) # (1, alpha_hat[:-1])
        self.alpha_hat_next = torch.concat((self.alpha_hat[1:], torch.zeros(1).to(self.alpha_hat)))
        
        # Compute posterior log variance, same shape as alpha and beta
        posterior_variance = self.beta * (1.0 - self.alpha_hat_prev) / (1.0 - self.alpha_hat)
        self.posterior_log_variance = torch.log(torch.concat((posterior_variance[1:2], posterior_variance[1:])))
        self.timesteps = torch.range(noise_steps-1, -1, -1).to(device)
        
        
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
    
    @torch.no_grad()
    def sample(self, model: nn.Module, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                
                # Generate the same timestep for all batch of images
                t = (torch.ones(n) * i).long().to(self.device)
                x = x.float()
                pred = model(x, t)
                pred_noise, pred_var = torch.split(pred, 3, dim=1)
                
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                log_variance = self.posterior_log_variance[t][:, None, None, None]
                
                # Compute the standard deviation from log variance
                if i > 1:
                    # interpolation between beta and beta_tilde
                    min_log = log_variance
                    max_log = torch.log(beta)
                    frac = (pred_var+1.0) / 2.0
                    log_variance = frac * max_log + (1.0 - frac) * min_log
                    stddev = torch.exp(0.5 * log_variance)
                else:
                    stddev = torch.zeros_like(beta)
                    
                    
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                    
                x = 1 / torch.sqrt(alpha) * (x - ((1-alpha) / (torch.sqrt(1-alpha_hat))) * pred_noise) + stddev * noise
        
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

# Training losses
def pred_mean_var(iddpm: IDDPM, pred_noise, pred_var, x_t, t):
    beta = iddpm.beta[t][:, None, None, None]
    alpha = iddpm.alpha[t][:, None, None, None]
    alpha_hat = iddpm.alpha_hat[t][:, None, None, None]
    log_variance = iddpm.posterior_log_variance[t][:, None, None, None]
    
    min_log = log_variance
    max_log = torch.log(beta)
    frac = (pred_var+1.0) / 2.0
    log_variance = frac * max_log + (1.0 - frac) * min_log
    mean = 1 / torch.sqrt(alpha) * (x_t - ((1-alpha) / (torch.sqrt(1-alpha_hat))) * pred_noise)
    return mean, log_variance

def true_mean_var(iddpm: IDDPM, x_0, x_t, t):
    beta = iddpm.beta[t][:, None, None, None]
    alpha = iddpm.alpha[t][:, None, None, None]
    alpha_hat = iddpm.alpha_hat[t][:, None, None, None]
    alpha_hat_prev = iddpm.alpha_hat_prev[t][:, None, None, None]
    log_variance = iddpm.posterior_log_variance[t][:, None, None, None]
    
    mean_coef1 = beta * alpha_hat_prev ** 0.5 / (1.0 - alpha_hat)
    mean_coef2 = (1.0 - alpha_hat_prev) * alpha ** 0.5 / (1.0 - alpha_hat)
    
    mean = mean_coef1 * x_0 + mean_coef2 * x_t
    return mean, log_variance

def KL_divergence(mean1: torch.Tensor, var1: torch.Tensor, mean2: torch.Tensor, var2: torch.Tensor):
    return 0.5 * (-1.0 + var2 - var1 + torch.exp(var1-var2) + ((mean1-mean2)**2) * torch.exp(-var2))

def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def gaussian_nll(clean_images: torch.Tensor, pred_mean: torch.Tensor, pred_logvar: torch.Tensor,):
    centered_x = clean_images - pred_mean
    inv_stdv = torch.exp(-pred_logvar)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in).clamp_min(1e-12)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in).clamp_min(1e-12)
    cdf_delta = (cdf_plus - cdf_min).clamp_min(1e-12)
    log_cdf_plus = torch.log(cdf_plus)
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp_min(1e-12))
    log_probs = torch.log(cdf_delta.clamp_min(1e-12))
    log_probs[clean_images < -0.999] = log_cdf_plus[clean_images < -0.999]
    log_probs[clean_images > 0.999] = log_one_minus_cdf_min[clean_images > 0.999]
    return log_probs

def vlb_loss(iddpm: IDDPM, pred_noise, pred_var, x_0, x_t, t):
    pred_mean, pred_log_var = pred_mean_var(iddpm, pred_noise, pred_var, x_t, t)
    true_mean, true_log_var = true_mean_var(iddpm, x_0, x_t, t)
    
    kl = KL_divergence(true_mean, true_log_var, pred_mean, pred_log_var)
    kl = kl.mean(dim=list(range(1, len(kl.shape)))) / math.log(2.0)
    
    nll = gaussian_nll(x_0, pred_mean, pred_log_var * 0.5)
    nll = nll.mean(dim=list(range(1, len(nll.shape)))) / math.log(2.0)
    res = torch.where(t == 0, nll, kl) # if t == 0, select ele in nll, else select ele in kl
    return res


def training_losses(iddpm: IDDPM, model: nn.Module, x_0: torch.Tensor, noise: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, vlb_lambda: float = 1e-3):
    x_t = x_t.float()
    pred = model(x_t, t)
    pred_noise, pred_var = torch.split(pred, 3, dim=1)
    mse = nn.MSELoss()
    loss_simple = mse(noise, pred_noise)
    loss_vlb = vlb_loss(iddpm, pred_noise.detach(), pred_var, x_0, x_t, t)
    return loss_simple + vlb_lambda * loss_vlb


# Importance Sampler
class ImportanceSampler:
    
    def __init__(self, diffusion_steps: int = 1000, history_per_term: int = 10):
        self.diffusion_steps = diffusion_steps
        self.history_per_term = history_per_term
        self.uni_prob = 1.0 / diffusion_steps
        self.loss_history = np.zeros([diffusion_steps, history_per_term], dtype=np.float64)
        self.loss_cnt = np.zeros([diffusion_steps], dtype=int)
        
    def sample(self, batch_size: int):
        weights = self.weights
        prob = weights / np.sum(weights)
        t = np.random.choice(self.diffusion_steps, size=(batch_size, ), p=prob)
        weights = 1.0 / (self.diffusion_steps * prob[t])
        return torch.from_numpy(t).long(), torch.from_numpy(weights).float()
        
    @property
    def weights(self):
        if not np.all(self.loss_cnt == self.history_per_term):
            return np.ones([self.diffusion_steps], dtype=np.float64)
        weights = np.sqrt(np.mean(self.loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1.0 - self.uni_prob
        weights += self.uni_prob / len(weights)
        return weights
    
    def update(self, t: torch.Tensor, losses: torch.Tensor):
        if dist.is_initialized():
            world_size = dist.get_world_size()
            # get batch sizes for padding to the maximum bs
            batch_sizes = [torch.tensor([0], dtype=torch.int32, device=t.device) for _ in range(world_size)]
            dist.all_gather(batch_sizes, torch.full_like(batch_sizes[0], t.size(0)))
            max_batch_size = max([bs.item() for bs in batch_sizes])
            # gather all timesteps and losses
            timestep_batches: List[torch.Tensor] = [torch.zeros(max_batch_size).to(t) for _ in range(world_size)]
            loss_batches: List[torch.Tensor] = [torch.zeros(max_batch_size).to(losses) for _ in range(world_size)]
            dist.all_gather(timestep_batches, t)
            dist.all_gather(loss_batches, losses)
            all_timesteps = [ts.item() for ts_batch, bs in zip(timestep_batches, batch_sizes) for ts in ts_batch[:bs]]
            all_losses = [loss.item() for loss_batch, bs in zip(loss_batches, batch_sizes) for loss in loss_batch[:bs]]
        else:
            all_timesteps = t.tolist()
            all_losses = losses.tolist()
            
        for t, loss in zip(all_timesteps, all_losses):
            if self.loss_cnt[t] == self.history_per_term:
                self.loss_history[t, :-1] = self.loss_history[t, 1:]
                self.loss_history[t, -1] = loss
            else:
                self.loss_history[t, self.loss_cnt[t]] = loss
                self.loss_cnt[t] += 1

def get_transform(args):
    preprocess = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    def transform(samples):
        images = [preprocess(img.convert("RGB")) for img in samples["image"]]
        return dict(images=images)
    return transform 
    
def train_model(args):
    setup_logging(args.run_name)
    device = args.device
    # dataloader = getData(args)
    
    dataset = load_dataset("huggan/few-shot-anime-face", split="train")
    # dataset = dataset.select(range(21551))
    dataset.set_transform(get_transform(args))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    iddpm = IDDPM()
    model = Variance_Unet().to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(dataloader) * args.epochs),
    )
    importance_sampler = ImportanceSampler()
    diffusion = IDDPM(img_size=args.img_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}: ")
        pbar = tqdm(dataloader)
        # for i, (imgs, _) in enumerate(pbar):
        for i, batch in enumerate(pbar):
            imgs = batch["images"]
            imgs = imgs.to(device)
            # t = diffusion.sample_timesteps(imgs.shape[0]).to(device)
            t, weights = importance_sampler.sample(imgs.shape[0])
            t = t.to(device)
            weights = weights.to(device)
            xt, noise = diffusion.noise_imgs(imgs, t)
            
            # xt = xt.float()
            # pred_noise = model(xt, t)
            # loss = mse(noise, pred_noise)
            losses = training_losses(iddpm, model, imgs, noise, xt, t)
            importance_sampler.update(t, losses)
            loss = (losses * weights).mean()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()
            
            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("Simple & VLB loss", loss.item(), global_step=epoch * l + i)
            
        sampledImgs = diffusion.sample(model, n = imgs.shape[0])
        saveImg(sampledImgs, os.path.join("results", args.run_name, f"{epoch}.png"))
        torch.save(model.state_dict(), os.path.join("ckpts", args.run_name, "{}.pt".format(args.run_name)))
            

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--run_name', type=str, default="IDDPM_butterfly", help='Model Type')
    parse.add_argument('--epochs', type=int, default=100)
    parse.add_argument('--batch_size', type=int, default=16)
    parse.add_argument('--img_size', type=int, default=64)
    parse.add_argument('--data_path', type=str, default="../../Dataset")
    parse.add_argument('--device', type=str, default="cuda")
    parse.add_argument('--lr', type=float, default=1e-4)
    parse.add_argument('--lr_warmup_steps', type=int, default=500)
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    train_model(args)
    