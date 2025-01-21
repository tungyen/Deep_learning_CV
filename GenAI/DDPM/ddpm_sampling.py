import torch
from utils import *
from model import Unet
from ddpm import *


if __name__ == '__main__':
    device = "cuda"
    model = Unet().to(device)
    ckpt = torch.load("unconditional_ckpt.pt")
    model.load_state_dict(ckpt)
    diffusion = Diffusion(img_size=64, device=device)
    x = diffusion.sample(model, n=16)
    plotImg(x)