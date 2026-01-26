from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import default_collate
import os
import torch

from Object_detection_2d.data.dataset import build_dataset, build_cmap
from Object_detection_2d.data.transforms import build_transforms, build_target_transform
from Object_detection_2d.data.container import Container

class BatchCollator:
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, batch):
        input_dict = Container(
            {key: default_collate([d[key] for d in batch]) for key in batch[0]}
        )
        return input_dict


def build_dataloader(opts):
        
    train_transform = build_transforms(opts.transforms.train)
    val_transform = build_transforms(opts.transforms.val)
    target_transform = build_target_transform(opts.target_transforms)
        
    train_dataset = build_dataset(opts.datasets.train, transform=train_transform, target_transform=target_transform)
    val_dataset = build_dataset(opts.datasets.val, transform=val_transform)
    test_dataset = build_dataset(opts.datasets.test, transform=val_transform)

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        
    train_dataloader = DataLoader(train_dataset, batch_size=opts.train_batch_size // world_size,
                                  sampler=train_sampler, collate_fn=BatchCollator(is_train=True), num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=opts.eval_batch_size // world_size,
                                sampler=val_sampler, collate_fn=BatchCollator(is_train=False), num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=opts.test_batch_size // world_size,
                                 sampler=test_sampler, collate_fn=BatchCollator(is_train=False))
    
    return train_dataloader, val_dataloader, test_dataloader