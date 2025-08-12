from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os

from Object_detection_2d.dataset.voc import VocDetectionDataset, voc_id2class
from Object_detection_2d.dataset.transforms import *

def get_dataset(args):
    dataset_type = args.dataset
    crop_size = args.crop_size
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    test_batch_size = args.test_batch_size
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
        
    if dataset_type == "voc":
        voc_data_root = args.voc_data_root
        voc_year = args.voc_year
        voc_download = args.voc_download
        crop_size = args.crop_size
        
        train_transform = Compose([
            ColorJitter(),
            ToTensor(),
            RandomExpand(mean),
            RandomCrop(),
            RandomHorizontalFlip(),
            Resize(size=crop_size),
            Normalize(mean=mean, std=std),
        ])
        
        val_transform = Compose([
            ToTensor(),
            Resize(size=crop_size),
            Normalize(mean=mean, std=std),
        ])
        
        train_dataset = VocDetectionDataset(root=voc_data_root, year=voc_year, split='train', download=voc_download, transform=train_transform)
        val_dataset = VocDetectionDataset(root=voc_data_root, year=voc_year, split='val', download=False, transform=val_transform, keep_difficult=True)
        test_dataset = VocDetectionDataset(root=voc_data_root, year=voc_year, split='val', download=False, transform=val_transform, keep_difficult=True)
        class_dict = voc_id2class
    else:
        raise ValueError(f'Unknown dataset {dataset_type}.')

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size // world_size,
                                  sampler=train_sampler, collate_fn=train_dataset.collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size // world_size,
                                sampler=val_sampler, collate_fn=val_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size // world_size,
                                 sampler=test_sampler, collate_fn=val_dataset.collate_fn)
    
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    
    return train_dataloader, val_dataloader, test_dataloader, class_dict, mean, std