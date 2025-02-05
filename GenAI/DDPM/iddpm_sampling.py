import torch
from utils import *
from iddpm import *


if __name__ == '__main__':
    device = "cuda"
    model = Variance_Unet().to(device)
    ckpt = torch.load("ckpts/IDDPM_butterfly/IDDPM_butterfly.pt")
    model.load_state_dict(ckpt)
    diffusion = IDDPM(img_size=64, device=device)
    x = diffusion.sample(model, n=16)
    saveImg(x, path="images/iddpm_res_butterfly.png")