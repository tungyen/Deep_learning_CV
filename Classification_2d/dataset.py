from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
import torch
from PIL import Image as im
import os
import json

def split_dataset_train_val(dataset: Dataset, split=0.9):
    train_size = int(split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

class flowerDataset(Dataset):
    def __init__(self, root, transform=None):
        
        flowerClasses = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
        flowerClasses.sort()
        classIndex = dict((key, value) for value, key in enumerate(flowerClasses))
        self.classIndex = classIndex
        jsonStr = json.dumps(dict((val, key) for key, val in classIndex.items()), indent=4)
        with open('classIndex.json', 'w') as jsonFile:
            jsonFile.write(jsonStr)
            
        json_path = 'classIndex.json'
        with open(json_path, "r") as f:
            self.class_indict = json.load(f)
        
        self.root = root
        self.transform = transform
        
        self.img_paths = []
        self.labels = []
        
        for cla in os.listdir(root):
            for path in os.listdir(os.path.join(root, cla)):
                self.img_paths.append(os.path.join(root, cla, path))
                self.labels.append(classIndex[cla])
                
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = im.open(self.img_paths[idx])
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
            
        return img, label
    
    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
    
def get_dataset(args):
    dataset_type = args.dataset
    batch_size = args.batch_size
    img_size = args.img_size
    if dataset_type == "flower":
        path = args.data_path
        data_transform = {
            "train": transforms.Compose([transforms.RandomResizedCrop(img_size), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            "val": transforms.Compose([transforms.Resize(256), transforms.CenterCrop(img_size), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
        train_path = os.path.join(path, "train")
        test_path = os.path.join(path, "val")
        
        train_dataset = flowerDataset(train_path, data_transform["train"])
        collate_fn = train_dataset.collate_fn
        class_dict = train_dataset.class_indict
        train_dataset, val_dataset = split_dataset_train_val(train_dataset)
        val_dataset.transform = data_transform["val"]
        test_dataset = flowerDataset(test_path, data_transform["val"])
        
        
    elif dataset_type == "cifar10":
        train_transform = transforms.Compose([transforms.Resize([img_size, img_size]),
                                            transforms.RandomCrop(img_size, padding=4), 
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandAugment(),
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
        train_dataset = datasets.CIFAR10(dataset_type, train=True, download=True, transform=train_transform)
        class_dict = {str(i): name for i, name in enumerate(train_dataset.classes)}
        train_dataset, val_dataset = split_dataset_train_val(train_dataset)
        
        val_transform = transforms.Compose([transforms.Resize([img_size, img_size]), 
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
        val_dataset.transform = val_transform
        test_dataset = datasets.CIFAR10(dataset_type, train=False, download=True, transform=val_transform)
        
    elif dataset_type == "cifar100":
        train_transform = transforms.Compose([transforms.Resize([img_size, img_size]),
                                            transforms.RandomCrop(img_size, padding=4), 
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandAugment(),
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
        train_dataset = datasets.CIFAR100(dataset_type, train=True, download=True, transform=train_transform)
        class_dict = {str(i): name for i, name in enumerate(train_dataset.classes)}
        train_dataset, val_dataset = split_dataset_train_val(train_dataset)
        
        val_transform = transforms.Compose([transforms.Resize([img_size, img_size]), 
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
        val_dataset.transform = val_transform
        test_dataset = datasets.CIFAR100(dataset_type, train=False, download=True, transform=val_transform)
    else:
        raise ValueError(f'unknown dataset {dataset_type}')
        
        
    if dataset_type == "flower":
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=test_dataset.collate_fn)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        
    return train_dataloader, val_dataloader, test_dataloader, class_dict

def get_dataset_stat(args):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    if args.dataset == "flower":
        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    
    return mean, std