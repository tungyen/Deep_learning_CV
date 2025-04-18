from model import *
import torch
from torchvision.utils import save_image
import argparse
import math
import os
import numpy as np

def sample(G: nn.Module, labels, B, latent_dim):
    z = torch.randn(B, latent_dim, 1, 1).to(args.device)
    x = G(z, labels)
    # x = x.view(-1, C, img_size, img_size)
    return x

def test_model(args):
    os.makedirs("img", exist_ok=True)
    device = args.device
    B = args.batch_size
    dataset_type = args.dataset
    latent_dim = args.latent_dim
    
    if dataset_type == "MNIST":
        C = 1
        class_num = 10
    elif dataset_type == "cifar":
        C = 3
        class_num = 10
    

    G = Generator(latent_dim, C, class_num).to(device)
    ckpts = torch.load("ckpts/Deep_Convolutional_GAN_G_{}.pt".format(dataset_type))
    G.load_state_dict(ckpts)
        
    G.eval()
    labels = torch.randint(0, class_num, size=(B, )).to(device)
    print(labels)
        
    x = sample(G, labels, B, latent_dim).detach().cpu()
    save_image(x, "img/Deep_Convolutional_GAN_{}_gen.png".format(dataset_type), nrow=int(math.sqrt(B)))


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', type=str, default="cifar")
    parse.add_argument('--latent_dim', type=int, default=100)
    parse.add_argument('--batch_size', type=int, default=16)
    parse.add_argument('--device', type=str, default="cuda")
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    test_model(args) 