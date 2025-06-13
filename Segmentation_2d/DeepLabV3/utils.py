import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler, StepLR
import pickle

from model import DeepLabV3

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [ max( base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power, self.min_lr)
                for base_lr in self.base_lrs]

def get_model(args):
    dataset_type = args.dataset
    model_name = args.model
    class_num = args.class_num
    device = args.device
    backbone = args.backbone
    momemtum = args.bn_momentum
    
    if dataset_type == "cityscapes":
        class_num += 1
    
    if model_name == "deeplabv3":
        model = DeepLabV3(class_num=class_num, in_channel=2048, backbone=backbone).to(device)
        set_bn_momentum(model.backbone, momentum=momemtum)
        return model
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
        
        # return nn.CrossEntropyLoss(ignore_index=class_num, weight=class_weights)
        return nn.CrossEntropyLoss(ignore_index=class_num)
    else:
        raise ValueError(f'Unknown dataset {dataset_type}')
    

def get_scheduler(args, optimizer):
    
    if args.scheduler == "poly":
        return PolyLR(optimizer, args.epochs)
    elif args.scheduler == "step":
        return StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    else:
        raise ValueError(f'Unknown scheduler {args.scheduler}')

def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum