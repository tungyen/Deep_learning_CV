import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy.stats import wasserstein_distance, entropy

from dataset import *
from model import *
import torch
device="cuda"
model = VAE_PI(5, 2).to(device)
ckpts = torch.load("ckpts/pi.pt")
model.load_state_dict(ckpts)

# Load real data
xs = np.load("pi_xs.npy")
ys = np.load("pi_ys.npy")
image_array = np.array(Image.open("sparse_pi_colored.jpg"))
rgb_values = image_array[xs, ys]
real_data = np.column_stack([xs, ys, rgb_values])

latent_dim = 2
img_size = 300
def generate_samples(batch_size):
    z = torch.randn(batch_size, latent_dim)
    z = z.to(device)
    reconstruct = model.decode(z)
    pos = reconstruct[:, :2]
    color = reconstruct[:, 2:]
    pos = torch.clamp(pos, min=0, max=(img_size-1)/img_size).cpu().detach().numpy()
    color = torch.clamp(color, min=0, max=1).cpu().detach().numpy()
    
    pos = np.floor(pos * img_size+0.5).astype(int)
    color = (color*255).astype(np.uint8)
    return np.column_stack([pos, color])

generated_data = generate_samples(5000)

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

# Compute the KL divergence
bins = np.histogram_bin_edges(real_data[:, 0], bins="auto")
kl_divs = [entropy(np.histogram(real_data[:, i], bins=bins, density=True)[0], 
                   np.histogram(generated_data[:, i], bins=bins, density=True)[0]) for i in range(5)]
print("KL divergences:", kl_divs)
