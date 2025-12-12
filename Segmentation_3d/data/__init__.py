from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os

from Segmentation_3d.data.dataset import build_dataset
from Segmentation_3d.data.transforms import build_transforms

def build_dataloader(opts):
    train_transforms = build_transforms(opts.transforms.train)
    val_transforms = build_transforms(opts.transforms.val)

    train_dataset = build_dataset(opts.datasets.train, transforms=train_transforms)
    val_dataset = build_dataset(opts.datasets.val, transforms=val_transforms)
    test_dataset = build_dataset(opts.datasets.test, transforms=val_transforms)

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        
    train_dataloader = DataLoader(train_dataset, batch_size=opts.train_batch_size // world_size,
                                  sampler=train_sampler, num_workers=2, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=opts.val_batch_size // world_size,
                                sampler=val_sampler, num_workers=2, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=opts.test_batch_size // world_size, sampler=test_sampler)
    
    return train_dataloader, val_dataloader, test_dataloader