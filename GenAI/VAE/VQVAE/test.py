from dataset import *
from model import *
import torch
from torchvision.utils import save_image
import argparse
import math

def sample(model: nn.Module, gen_img_size, args):
    z = torch.randn(args.sample_batch_size, args.emb_dim, gen_img_size, gen_img_size).to(args.device)
    z, _ = model.vq_layer(z)
    x = model.decode(z)
    x = x.view(-1, 3, args.img_size, args.img_size)
    return x

def test_model(args):
    task = args.task
    device = args.device
    model = VQVAE(3, args.emb_num, args.emb_dim).to(device)
    ckpts = torch.load(args.ckpts_path)
    model.load_state_dict(ckpts)
    dataset = CelebA_dataset(args.root)
    dataloader = DataLoader(dataset, batch_size=args.sample_batch_size, shuffle=True)
    
    tmp = torch.zeros((args.batch_size, 3, args.img_size, args.img_size)).to(device)
    output = model.encode(tmp)
    output, _ = model.vq_layer(output)
    gen_img_size = output.shape[2]
    
    if task == 'reconstruct':
        model.eval()
        batch = next(iter(dataloader)).to(device)
        [reconstruct, _] = model(batch)
        reconstruct = reconstruct.view(-1, 3, args.img_size, args.img_size)
        combined = torch.cat((batch, reconstruct)).detach().cpu()
        save_image(combined, "img/reconst_res.png", nrow=args.sample_batch_size)
        
    elif task == 'generate':
        model.eval()
        x = sample(model, gen_img_size, args).detach().cpu()
        save_image(x, "img/gen_res.png", nrow=int(math.sqrt(args.sample_batch_size)))

    else:
        raise ValueError(f'unknown test task {task}')


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--task', type=str, default="generate")
    parse.add_argument('--emb_num', type=int, default=512)
    parse.add_argument('--emb_dim', type=int, default=64)
    parse.add_argument('--batch_size', type=int, default=16)
    parse.add_argument('--sample_batch_size', type=int, default=16)
    parse.add_argument('--img_size', type=int, default=64)
    parse.add_argument('--device', type=str, default="cuda")
    parse.add_argument('--root', type=str, default="../../../Dataset/img_align_celeba")
    parse.add_argument('--ckpts_path', type=str, default="ckpts/vqvae.pt")
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    test_model(args) 