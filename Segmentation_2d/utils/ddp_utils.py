import torch.nn as nn
import torch
import torch.distributed as dist

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def all_reduce_confusion_matrix(confusion_matrix, local_rank):
    tensor = confusion_matrix.confusion_matrix.to(torch.device(f"cuda:{local_rank}"))
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    confusion_matrix.confusion_matrix = tensor

def is_main_process():
    return get_rank() == 0