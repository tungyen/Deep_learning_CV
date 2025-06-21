from dataset import *
from model import *
import torch
from torchvision.utils import save_image
import argparse
import math
from torchvision import datasets


def test_model(args):
    os.makedirs("imgs", exist_ok=True)
    task = args.task
    device = args.device
    B = args.batch_size
    K = args.emb_num
    dataset_type = args.dataset
    
    if dataset_type == "celeba":
        dataset = CelebA_dataset("../../../Dataset/img_align_celeba")
        dataloader = DataLoader(dataset, batch_size=B, shuffle=True)
        C = 3
        class_num = None
        img_size = 64
        
    elif dataset_type == "MNIST":
        dataloader = DataLoader(datasets.MNIST('data', train=False, transform=transforms.ToTensor()), batch_size=B)
        C = 1
        class_num = 10
        img_size = 28
        
    elif dataset_type == 'fashion':
        dataloader = DataLoader(datasets.FashionMNIST('fashion', train=False, transform=transforms.ToTensor()), batch_size=B)
        C = 1
        class_num = 10
        img_size = 28
    elif dataset_type == 'cifar':
        dataloader = DataLoader(datasets.CIFAR10('cifar', train=False, transform=transforms.ToTensor()), batch_size=B)
        C = 3
        class_num = 10
        img_size = 32
    
    model = VQVAE(C, K, args.emb_dim, args.n_block, class_num, args.prior_dim).to(device)
    
    if task == 'recon':
        ckpts = torch.load("ckpts/VQVAE_{}.pt".format(dataset_type))
        model.load_state_dict(ckpts)
        model.eval()
        batch = next(iter(dataloader))
        
        if dataset_type == "celeba":
            img = batch.to(device)
            label = None
        elif dataset_type == "MNIST":
            img, label = batch
            img = img.to(device)
            label = label.to(device)
        elif dataset_type == "fashion":
            img, label = batch
            img = img.to(device)
            label = label.to(device)
        elif dataset_type == "cifar":
            img, label = batch
            img = img.to(device)
            label = label.to(device)
        
        [reconstruct, _] = model(img, label)
        reconstruct = reconstruct.view(-1, C, img_size, img_size)
        combined = torch.cat((img, reconstruct)).detach().cpu()
        save_image(combined, "img/VQVAE_{}_recon.png".format(dataset_type), nrow=B)
        
    elif task == 'gen':
        ckpts = torch.load("ckpts/VQVAE_prior_{}.pt".format(dataset_type))
        model.load_state_dict(ckpts)
        model.eval()
        
        if dataset_type == "celeba":
            label = None
        else:
            label = torch.randint(0, class_num, size=(B, )).to(device)
            print(label)
        H, W = img_size // 4, img_size // 4
        q_sample = torch.zeros((B, H, W)).long().to(device)
        
        for i in range(H):
            for j in range(W):
                print("Currently running on {} and {}".format(i, j))
                output = model.pixelcnn_prior(q_sample, label)
                probs = F.softmax(output[:, :, i, j], dim=-1)
                q_sample.data[:, i, j].copy_(probs.multinomial(1).data.squeeze())
                
        z_sample = model.vq_layer.emb.weight[q_sample].permute(0, 3, 1, 2)
        x_sample = model.decode(z_sample)
        x_sample = x_sample.view(-1, C, img_size, img_size)
        save_image(x_sample, "img/VQVAE_{}_gen.png".format(dataset_type), nrow=int(math.sqrt(B)))
    else:
        raise ValueError(f'unknown test task {task}')


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', type=str, default="celeba")
    parse.add_argument('--task', type=str, default="gen")
    parse.add_argument('--emb_num', type=int, default=512)
    parse.add_argument('--emb_dim', type=int, default=64)
    parse.add_argument('--n_block', type=int, default=15)
    parse.add_argument('--prior_dim', type=int, default=256)
    parse.add_argument('--batch_size', type=int, default=16)
    parse.add_argument('--device', type=str, default="cuda")
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    test_model(args) 