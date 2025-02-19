import torch
from utils import *
from unet import UNet
from ddim import *

ckpt_name = "DDPM_landscape"
runname = "DDIM_landscape"
if __name__ == '__main__':
    device = "cuda"
    model = UNet().to(device)
    ckpt = torch.load(os.path.join("ckpts", ckpt_name, ckpt_name+".pt"))
    model.load_state_dict(ckpt)
    diffusion = DDIM(img_size=64, device=device)
    x = diffusion.sample(model, n=16)
    saveImg(x, path=os.path.join("images", runname+".png"))