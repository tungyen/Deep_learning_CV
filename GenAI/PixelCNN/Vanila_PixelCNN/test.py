from model2 import *
import torch
from torchvision.utils import save_image
import argparse
import math
import torch.nn.functional as F


def test_model(args):
    if args.datasets == 'MNIST':
        C = 1
        H, W = 28, 28
    elif args.datasets == 'CIFAR':
        C = 3
        H, W = 32
    else:
        raise ValueError(f'unknown dataset {args.datasets}')
    
    ckpts_path = 'ckpts/pixelCnn2_{}_{}.pt'.format(args.datasets ,args.color_level)
    device = args.device
    n_block = args.n_block
    h_dim = args.h_dim
    B = args.batch_size
    color_level = args.color_level
    
    # model = PixelCNN(C, n_block, h_dim).to(device)
    model = PixelCNN(C, color_level, n_block, h_dim).to(device)
    ckpts = torch.load(ckpts_path)
    model.load_state_dict(ckpts)
    
    imgs = torch.zeros((B, C, H, W)).to(device)
    with torch.no_grad():
        for i in range(H):
            for j in range(W):
                pred = model(imgs)
                prob = F.softmax(pred[:, :, :, i, j], dim=1)
                for k in range(C):
                    pixel = torch.multinomial(prob[:, :, k], 1).float() / (color_level - 1)
                    pixel = pixel.squeeze()
                    imgs[:, k, i, j] = pixel

    imgs = imgs.clamp(0, 1.0)
    rows = int(math.sqrt(B))
    save_image(imgs, "img/pixelCnn_{}_{}.png".format(args.datasets, color_level), nrow=rows)

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--datasets', type=str, default='MNIST')
    parse.add_argument('--batch_size', type=int, default=16)
    parse.add_argument('--device', type=str, default="cuda")
    parse.add_argument('--n_block', type=int, default=8)
    parse.add_argument('--h_dim', type=int, default=64)
    parse.add_argument('--color_level', type=int, default=256)
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    test_model(args) 