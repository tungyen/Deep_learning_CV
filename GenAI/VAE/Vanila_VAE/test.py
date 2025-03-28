from dataset import *
from model import *
import torch
from torchvision.utils import save_image
import argparse
import math

def sample(model: nn.Module, args):
    z = torch.randn(args.sample_batch_size, args.latent_dim).to(args.device)
    x = model.decode(z)
    x = x.view(-1, 3, args.img_size, args.img_size)
    return x

def test_model(args):
    task = args.task
    device = args.device
    model = VAE(3, args.latent_dim).to(device)
    ckpts = torch.load(args.ckpts_path)
    model.load_state_dict(ckpts)
    dataset = CelebA_dataset(args.root)
    dataloader = DataLoader(dataset, batch_size=args.sample_batch_size, shuffle=True)
    model_name = "VAE"
    if task == 'reconstruct':
        model.eval()
        batch = next(iter(dataloader)).to(device)
        [reconstruct, _, _] = model(batch)
        reconstruct = reconstruct.view(-1, 3, args.img_size, args.img_size)
        combined = torch.cat((batch, reconstruct)).detach().cpu()
        save_image(combined, "img/{}_reconst.png".format(model_name), nrow=args.sample_batch_size)
        
    elif task == 'generate':
        model.eval()
        x = sample(model, args).detach().cpu()
        save_image(x, "img/{}_gen.png".format(model_name), nrow=int(math.sqrt(args.sample_batch_size)))

    else:
        raise ValueError(f'unknown test task {task}')


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--task', type=str, default="generate")
    parse.add_argument('--latent_dim', type=int, default=128)
    parse.add_argument('--batch_size', type=int, default=16)
    parse.add_argument('--sample_batch_size', type=int, default=16)
    parse.add_argument('--img_size', type=int, default=64)
    parse.add_argument('--device', type=str, default="cuda")
    parse.add_argument('--root', type=str, default="../../../Dataset/img_align_celeba")
    parse.add_argument('--ckpts_path', type=str, default="ckpts/vae.pt")
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    test_model(args) 