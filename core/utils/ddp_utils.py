import torch.nn as nn
import torch
import torch.distributed as dist

def gather_list(data):
    world_size = dist.get_world_size()
    data_list = [None for _ in range(world_size)]
    dist.all_gather_object(data_list, data)
    gathered_data = []
    for part in data_list:
        gathered_data.extend(part)
    return gathered_data

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def synchronize():
    """
       Helper function to synchronize (barrier) among all processes when
       using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def all_gather_object_safe(obj):
    world_size = dist.get_world_size()
    out_list = [None for _ in range(world_size)]
    dist.all_gather_object(out_list, obj)
    return out_list

def gather_dict(preds):
    all_predictions = all_gather_object_safe(preds)
    if not is_main_process():
        return []

    preds = {}
    for p in all_predictions:
        preds.update(p)
    img_ids = list(sorted(preds.keys()))
    preds = [preds[i] for i in img_ids]
    return preds