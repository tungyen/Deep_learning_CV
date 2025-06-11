import numpy as np
import torch
import torch.nn as nn
import pickle

from model.deeplabv3 import DeepLabV3

def get_model(args):
    dataset_type = args.dataset
    model_name = args.model
    class_num = args.class_num
    device = args.device
    
    if dataset_type == "cityscapes":
        class_num += 1
    
    if model_name == "deeplabv3":
        return DeepLabV3(class_num=class_num).to(device)
    else:
        raise ValueError(f'Unknown model {model_name}')
    
def get_criterion(args):
    dataset_type = args.dataset
    class_num = args.class_num
    device = args.device
    
    if dataset_type == "cityscapes":
        with open("../../Dataset/cityscapes/meta/class_weights.pkl", "rb") as file:
            class_weights = np.array(pickle.load(file))
        class_weights = torch.from_numpy(class_weights)
        class_weights = class_weights.type(torch.FloatTensor).to(device)
        
        return nn.CrossEntropyLoss(ignore_index=class_num, weight=class_weights)
    else:
        raise ValueError(f'Unknown dataset {dataset_type}')

def add_weight_decay(model, weight_decay, skip_list=()):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': weight_decay}]