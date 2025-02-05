import torch
from utils import *
from model import Unet
from ddim import *


if __name__ == '__main__':
    device = "cuda"
    model = Unet().to(device)
    ckpt = torch.load("ckpts/DDPM_butterfly/DDPM_butterfly.pt")
    model.load_state_dict(ckpt)
    diffusion = DDIM(img_size=64, device=device, sample_steps=100)
    x = diffusion.sample(model, n=16, eta=1.0)
    saveImg(x, path="images/ddim_res_butterfly.png")