from model import *
import torch
from torchvision.utils import save_image
import argparse
import math
import os

def sample(G: nn.Module, labels, B, img_size, C, latent_dim):
    z = torch.randn(B, latent_dim).to(args.device)
    x = G(z, labels)
    x = x.view(-1, C, img_size, img_size)
    return x

def test_model(args):
    os.makedirs("img", exist_ok=True)
    device = args.device
    B = args.batch_size
    dataset_type = args.dataset
    latent_dim = args.latent_dim
    
    if dataset_type == "SVHN":
        C = 3
        class_num = 10
        img_size = 32
        img_shape = np.array([C, img_size, img_size])
    elif dataset_type == "MNIST":
        C = 1
        class_num = 10
        img_size = 28
        img_shape = np.array([C, img_size, img_size])
    

    G = Generator(latent_dim, class_num, img_shape).to(device)
    ckpts = torch.load("ckpts/Conditional_GAN_G_{}.pt".format(dataset_type))
    G.load_state_dict(ckpts)
        
    G.eval()
    labels = torch.randint(0, class_num, size=(B, )).to(device)
    print(labels)
        
    x = sample(G, labels, B, img_size, C, latent_dim).detach().cpu()
    save_image(x, "img/Conditional_GAN_{}_gen.png".format(dataset_type), nrow=int(math.sqrt(B)))


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', type=str, default="MNIST")
    parse.add_argument('--latent_dim', type=int, default=100)
    parse.add_argument('--batch_size', type=int, default=16)
    parse.add_argument('--device', type=str, default="cuda")
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    test_model(args) 