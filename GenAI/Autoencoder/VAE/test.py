from dataset import *
from model import *
import torch
from torchvision.utils import save_image
import argparse
import math
from torchvision import datasets

def sample(model: nn.Module, labels, B, img_size, C, latent_dim):
    z = torch.randn(B, latent_dim).to(args.device)
    x = model.decode(z, labels)
    x = x.view(-1, C, img_size, img_size)
    return x

def test_model(args):
    task = args.task
    device = args.device
    B = args.batch_size
    dataset_type = args.dataset
    # latent_dim = args.latent_dim
    
    if dataset_type == "celeba":
        dataset = CelebA_dataset("../../../Dataset/img_align_celeba")
        dataloader = DataLoader(dataset, batch_size=B, shuffle=True)
        C = 3
        class_num = None
        img_size = 64
        latent_dim = 128
    elif dataset_type == "MNIST":
        dataloader = DataLoader(datasets.MNIST('data', train=False, transform=transforms.ToTensor()), batch_size=B)
        C = 1
        class_num = 10
        img_size = 28
        latent_dim = 10
    elif dataset_type == 'fashion':
        dataloader = DataLoader(datasets.FashionMNIST('fashion', train=False, transform=transforms.ToTensor()), batch_size=B)
        C = 1
        class_num = 10
        img_size = 28
        latent_dim = 10
    elif dataset_type == 'cifar':
        dataloader = DataLoader(datasets.CIFAR10('cifar', train=False, transform=transforms.ToTensor()), batch_size=B)
        C = 3
        class_num = 10
        img_size = 32
        latent_dim = 128
    
    
    
    model = VAE(C, latent_dim, class_num, img_size=img_size).to(device)
    ckpts = torch.load("ckpts/VAE_{}.pt".format(dataset_type))
    model.load_state_dict(ckpts)

    if task == 'recon':
        model.eval()
        batch = next(iter(dataloader))
        
        if dataset_type == "celeba":
            img = batch.to(device)
            label = None
        else:
            img, label = batch
            img = img.to(device)
            label = label.to(device)
        
        [reconstruct, _, _] = model(img, label)
        reconstruct = reconstruct.view(-1, C, img_size, img_size)
        combined = torch.cat((img, reconstruct)).detach().cpu()
        save_image(combined, "img/VAE_{}_recon.png".format(dataset_type), nrow=B)
        
    elif task == 'gen':
        model.eval()
        if class_num is None:
            labels = None
        else:
            labels = torch.randint(0, class_num, size=(B, )).to(device)
            print(labels)
            
        x = sample(model, labels, B, img_size, C, latent_dim).detach().cpu()
        save_image(x, "img/VAE_{}_gen.png".format(dataset_type), nrow=int(math.sqrt(B)))

    else:
        raise ValueError(f'unknown test task {task}')


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', type=str, default="cifar")
    parse.add_argument('--task', type=str, default="gen")
    parse.add_argument('--latent_dim', type=int, default=128)
    parse.add_argument('--batch_size', type=int, default=16)
    parse.add_argument('--device', type=str, default="cuda")
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    test_model(args) 