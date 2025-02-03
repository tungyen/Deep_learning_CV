import torch
from utils import *
from model import Conditional_Unet
from cfdg import *

if __name__ == '__main__':
    n = 10
    device = "cuda"
    model = Conditional_Unet(classNum=10).to(device)
    ckpt = torch.load("pretrained/conditional_ema_ckpt.pt")
    model.load_state_dict(ckpt)
    diffusion = CFDG(img_size=64, device=device)
    y = torch.Tensor([6] * n).long().to(device)
    x = diffusion.sample(model, n, y, scale=3)
    saveImg(x, path="images/cfdg_res.png")