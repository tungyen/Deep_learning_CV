import torch
import torch.distributed as dist

def gather_all_data(data):
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