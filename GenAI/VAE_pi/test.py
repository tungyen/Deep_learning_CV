from dataset import *
from model import *
import torch
from PIL import Image
import argparse

def test_model_img(args):
    device = args.device
    model = VAE_PI(5, args.latent_dim).to(device)
    ckpts = torch.load(args.ckpts_path)
    model.load_state_dict(ckpts)
    batch_size = args.batch_size

    img_size = args.img_size
    latent_dim = args.latent_dim

    times = args.loops
    img = np.zeros((img_size, img_size, 3)).astype(np.uint8)

    for _ in range(times):
        z = torch.randn(args.batch_size, args.latent_dim)
        z = z.to(device)
        reconstruct = model.decode(z)
        pos = reconstruct[:, :2]
        color = reconstruct[:, 2:]
        pos = torch.clamp(pos, min=0, max=(args.img_size-1)/args.img_size).cpu().detach().numpy()
        color = torch.clamp(color, min=0, max=1).cpu().detach().numpy()
        for i in range(reconstruct.shape[0]):
            p = pos[i]
            c = color[i]
            p = np.floor(p * args.img_size+0.5).astype(int)
            p = tuple(p)
            c = (c*255).astype(np.uint8)
            img[p[1], p[0], :] = c
            
    img = Image.fromarray(img)
    img.save("img/gen_pi.png")
        
def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--latent_dim', type=int, default=2)
    parse.add_argument('--batch_size', type=int, default=16)
    parse.add_argument('--img_size', type=int, default=300)
    parse.add_argument('--loops', type=int, default=1000)
    parse.add_argument('--device', type=str, default="cuda")
    parse.add_argument('--xs_path', type=str, default="pi_xs.npy")
    parse.add_argument('--ys_path', type=str, default="pi_ys.npy")
    parse.add_argument('--img_path', type=str, default="sparse_pi_colored.jpg")
    parse.add_argument('--ckpts_path', type=str, default="ckpts/pi.pt")
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    test_model_img(args)    