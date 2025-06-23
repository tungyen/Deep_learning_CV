import torch
import torch.nn as nn
from Segmentation_3d.PointNet.model import *

class FocalLoss(nn.Module):
    def __init__(self, numClasses=4, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.numClasses = numClasses
        self.gamma = gamma
        
    def forward(self, prediction, annotation):
        loss = nn.CrossEntropyLoss()
        CE_loss = loss(prediction, annotation)
        pt = torch.exp(-CE_loss)
        focalLoss = self.alpha * (1-pt) ** self.gamma * CE_loss
        return focalLoss
    
    
def get_model(args) -> nn.Module:
    model_name = args.model
    device = args.device
    class_num = args.class_num
    
    if model_name == "pointnet_cls":
        model = PointNetCls(class_num=class_num).to(device)
    elif model_name == "pointnet_seg":
        model = PointNetSeg(class_num=class_num).to(device)
    else:
        raise ValueError(f'unknown model {model_name}')
    
    return model

def get_loss(args):
    loss_func = args.loss_func
    
    if loss_func == "ce":
        return nn.CrossEntropyLoss()
    elif loss_func == "focal":
        return FocalLoss()
    else:
        raise ValueError(f'Unknown loss function {loss_func}.')