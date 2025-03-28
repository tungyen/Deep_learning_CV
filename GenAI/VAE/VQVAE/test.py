from dataset import *
from model import *
import torch
from torchvision.utils import save_image
import argparse
import math


def test_model(args):
    task = args.task
    device = args.device
    img_size = args.img_size
    B = args.batch_size
    K = args.emb_num
    
    model = VQVAE(3, args.emb_num, args.emb_dim, args.n_block, args.prior_dim).to(device)
    ckpts = torch.load(args.ckpts_path)
    model.load_state_dict(ckpts)
    dataset = CelebA_dataset(args.root)
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True)
    model_name = "VQVAE"
    model.eval()
    if task == 'reconstruct':
        batch = next(iter(dataloader)).to(device)
        [reconstruct, _] = model(batch)
        reconstruct = reconstruct.view(-1, 3, img_size, img_size)
        combined = torch.cat((batch, reconstruct)).detach().cpu()
        save_image(combined, "img/{}_reconst.png".format(model_name), nrow=B)
        
    elif task == 'generate':
        H, W = img_size, img_size
        q_sample = torch.zeros((B, H, W)).long().to(device)
        
        for i in range(H):
            for j in range(W):
                print("Currently running on {} and {}".format(i, j))
                output = model.pixelcnn_prior(q_sample)
                probs = F.softmax(output, dim=1)
                q_sample[:, i, j] = torch.multinomial(probs[:, :, i, j], 1).squeeze().float()
                
        q_sample = q_sample.reshape(-1, 1)
        q_onehot = torch.zeros(q_sample.shape[0], K, device=device)
        q_onehot.scatter_(1, q_sample, 1) # (BHW, emb_num)
        
        z_sample = torch.matmul(q_onehot, model.vq_layer.emb.weight) # (BHW, emb_dim)
        z_sample = z_sample.view(B, H, W, -1)
        z_sample = z_sample.permute(0, 3, 1, 2).contiguous()
        
        x_sample = model.decode(z_sample)
        x_sample = x_sample.view(-1, 3, img_size, img_size)
        save_image(x_sample, "img/{}_gen.png".format(model_name), nrow=int(math.sqrt(B)))

    else:
        raise ValueError(f'unknown test task {task}')


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--task', type=str, default="generate")
    parse.add_argument('--emb_num', type=int, default=512)
    parse.add_argument('--emb_dim', type=int, default=64)
    parse.add_argument('--n_block', type=int, default=15)
    parse.add_argument('--prior_dim', type=int, default=256)
    parse.add_argument('--batch_size', type=int, default=16)
    parse.add_argument('--sample_batch_size', type=int, default=16)
    parse.add_argument('--img_size', type=int, default=64)
    parse.add_argument('--device', type=str, default="cuda")
    parse.add_argument('--root', type=str, default="../../../Dataset/img_align_celeba")
    parse.add_argument('--ckpts_path', type=str, default="ckpts/VQVAE_prior.pt")
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    test_model(args) 