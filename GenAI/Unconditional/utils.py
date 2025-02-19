import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms

def plotImg(imgs):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([torch.cat([i for i in imgs.cpu()], dim=-1),], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

def saveImg(imgs, path, **kwargs):
    grid = torchvision.utils.make_grid(imgs, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)
    
def get_transform(args):
    preprocess = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    def transform(samples):
        images = [preprocess(img.convert("RGB")) for img in samples["image"]]
        return dict(images=images)
    return transform
    
def getData(args):
    if args.dataset == "landscape" or args.dataset == "cifar-10":
        landscape_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
            torchvision.transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = torchvision.datasets.ImageFolder(args.data_path, transform=landscape_transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        return dataloader
    elif args.dataset == "butterfly":
        dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
        dataset.set_transform(get_transform(args))
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        return dataloader
    else:
        raise ValueError(f'unknown dataset {args.dataset}')

def setup_logging(run_name):
    os.makedirs("ckpts", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("ckpts", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)