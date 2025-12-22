from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torch
from PIL import Image as im
import os
import json
    
def get_dataset(args):
    if dataset_type == "cifar10":
        train_transform = transforms.Compose([transforms.Resize([img_size, img_size]),
                                            transforms.RandomCrop(img_size, padding=4), 
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandAugment(),
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
        train_dataset = datasets.CIFAR10(os.path.join("Dataset", dataset_type), train=True, download=True, transform=train_transform)
        class_dict = {str(i): name for i, name in enumerate(train_dataset.classes)}
        train_dataset, val_dataset = split_dataset_train_val(train_dataset)
        
        val_transform = transforms.Compose([transforms.Resize([img_size, img_size]), 
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
        val_dataset.transform = val_transform
        test_dataset = datasets.CIFAR10(os.path.join("Dataset", dataset_type), train=False, download=True, transform=val_transform)
        
    elif dataset_type == "cifar100":
        train_transform = transforms.Compose([transforms.Resize([img_size, img_size]),
                                            transforms.RandomCrop(img_size, padding=4), 
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandAugment(),
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
        train_dataset = datasets.CIFAR100(os.path.join("Dataset", dataset_type), train=True, download=True, transform=train_transform)
        class_dict = {str(i): name for i, name in enumerate(train_dataset.classes)}
        train_dataset, val_dataset = split_dataset_train_val(train_dataset)
        
        val_transform = transforms.Compose([transforms.Resize([img_size, img_size]), 
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
        val_dataset.transform = val_transform
        test_dataset = datasets.CIFAR100(os.path.join("Dataset", dataset_type), train=False, download=True, transform=val_transform)
    else:
        raise ValueError(f'unknown dataset {dataset_type}')