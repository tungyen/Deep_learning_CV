import torch
from utils import *
from model import Unet
from ddim import *


if __name__ == '__main__':
    
    device = "cuda"
    model = Unet().to(device)
    ckpt = torch.load("pretrained/unconditional_ckpt.pt")
    model.load_state_dict(ckpt)
    diffusion = DDIM(img_size=64, device=device)
    x = diffusion.sample(model, n=16)
    saveImg(x, path="images/ddim_res.png")