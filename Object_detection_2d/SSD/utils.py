import json
import os
import torch
import random

def decimate(tensor, m):
    assert tensor.dim() == len(m)
    
    for d in range(len(m)):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())
            
    return tensor