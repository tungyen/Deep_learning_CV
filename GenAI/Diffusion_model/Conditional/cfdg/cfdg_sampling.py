import torch
import sys
import argparse

sys.path.append("..")
from utils import *
from unet import Conditional_UNet
from cfdg import *


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--img_num', type=int, default=10)
    parse.add_argument('--sampler', type=str, default="DDIM")
    parse.add_argument('--label', type=int, default=0)
    parse.add_argument('--img_size', type=int, default=64)
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    device = "cuda"
    model = Conditional_UNet(class_num=10).to(device)
    ckpt = torch.load("ckpts/CFDG_cifar/ema_ckpt.pt")
    model.load_state_dict(ckpt)
    diffusion = CFDG(img_size=64, device=device)
    y = torch.Tensor([args.label] * args.img_num).long().to(device)
    if args.sampler == "DDPM":
        x = diffusion.sample(model, args.img_num, y, scale=3)
    elif args.sampler == "DDIM":
        x = diffusion.sample_ddim(model, args.img_num, y, scale=3)
    else:
        raise ValueError(f'unknown diffusion sampler {args.sampler}')
        
    saveImg(x, path="images/CFDG_{}_cifar_{}.png".format(args.sampler, args.label))