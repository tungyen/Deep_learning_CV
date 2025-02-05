import torch
from utils import *
from model import Unet
from ddpm import *


if __name__ == '__main__':
    device = "cuda"
    model = Unet().to(device)
    ckpt = torch.load("ckpts/DDPM_butterfly/DDPM_butterfly.pt")
    model.load_state_dict(ckpt)
    diffusion = DDPM(img_size=64, device=device)
    x = diffusion.sample(model, n=16)
    saveImg(x, path="images/ddpm_res_butterfly.png")