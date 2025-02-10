import torch
from utils import *
from unet import UNet
from ddpm import *

runname = "DDPM_butterfly"
if __name__ == '__main__':
    device = "cuda"
    model = UNet().to(device)
    ckpt = torch.load(os.path.join("ckpts", runname, runname+".pt"))
    model.load_state_dict(ckpt)
    diffusion = DDPM(img_size=64, device=device)
    x = diffusion.sample(model, n=16)
    saveImg(x, path=os.path.join("images", runname+".png"))