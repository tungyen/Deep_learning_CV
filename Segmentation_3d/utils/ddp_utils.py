import torch
import torch.distributed as dist

def all_reduce_confusion_matrix(confusion_matrix, local_rank):
    tensor = confusion_matrix.confusion_matrix.to(torch.device(f"cuda:{local_rank}"))
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    confusion_matrix.confusion_matrix = tensor

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