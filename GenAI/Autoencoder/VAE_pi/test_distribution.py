import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy.stats import wasserstein_distance, entropy

from dataset import *
from model import *
import torch
import argparse


def test_model_distribution(args):
    device = args.device
    model = VAE_PI(5, args.latent_dim).to(device)
    ckpts = torch.load(args.ckpts_path)
    model.load_state_dict(ckpts)

    # Load real data
    xs = np.load(args.xs_path)
    ys = np.load(args.ys_path)
    image_array = np.array(Image.open(args.img_path))
    rgb_values = image_array[xs, ys]
    real_data = np.column_stack([xs, ys, rgb_values])
    
    def generate_samples(args):
        z = torch.randn(args.batch_size, args.latent_dim)
        z = z.to(device)
        reconstruct = model.decode(z)
        pos = reconstruct[:, :2]
        color = reconstruct[:, 2:]
        pos = torch.clamp(pos, min=0, max=(args.img_size-1)/args.img_size).cpu().detach().numpy()
        color = torch.clamp(color, min=0, max=1).cpu().detach().numpy()
        
        pos = np.floor(pos * args.img_size+0.5).astype(int)
        color = (color*255).astype(np.uint8)
        return np.column_stack([pos, color])

    generated_data = generate_samples(args)

    for i, label in enumerate(["x", "y", "r", "g", "b"]):
        plt.figure(figsize=(6, 4))
        sns.kdeplot(real_data[:, i], label="Real", fill=True)
        sns.kdeplot(generated_data[:, i], label="Generated", fill=True)
        plt.title(f"Distribution of {label}")
        plt.legend()
        plt.savefig('img/distribute_{}.png'.format(label), dpi=300, bbox_inches='tight')
        plt.show()

    # Compute the Wasserstein distance
    distances = [wasserstein_distance(real_data[:, i], generated_data[:, i]) for i in range(5)]
    print("Wasserstein distances:", distances)


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--latent_dim', type=int, default=2)
    parse.add_argument('--batch_size', type=int, default=5000)
    parse.add_argument('--img_size', type=int, default=300)
    parse.add_argument('--device', type=str, default="cuda")
    parse.add_argument('--xs_path', type=str, default="pi_xs.npy")
    parse.add_argument('--ys_path', type=str, default="pi_ys.npy")
    parse.add_argument('--img_path', type=str, default="sparse_pi_colored.jpg")
    parse.add_argument('--ckpts_path', type=str, default="ckpts/pi.pt")
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    test_model_distribution(args)
