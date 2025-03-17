from dataset import *
from model import *
import torch
from PIL import Image

device="cuda"
model = VAE_PI(5, 2).to(device)
ckpts = torch.load("ckpts/pi.pt")
model.load_state_dict(ckpts)
batch_size = 16

img_size = 300
latent_dim = 2


times = 1000
img = np.zeros((img_size, img_size, 3)).astype(np.uint8)

for _ in range(times):
    z = torch.randn(batch_size, latent_dim)
    z = z.to(device)
    reconstruct = model.decode(z)
    pos = reconstruct[:, :2]
    color = reconstruct[:, 2:]
    pos = torch.clamp(pos, min=0, max=(img_size-1)/img_size).cpu().detach().numpy()
    color = torch.clamp(color, min=0, max=1).cpu().detach().numpy()
    for i in range(reconstruct.shape[0]):
        p = pos[i]
        c = color[i]
        p = np.floor(p * img_size+0.5).astype(int)
        p = tuple(p)
        c = (c*255).astype(np.uint8)
        img[p[1], p[0], :] = c
        
img = Image.fromarray(img)
img.save("gen_pi.png")
        
        