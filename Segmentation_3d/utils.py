import torch
import torch.nn as nn
from Segmentation_3d.PointNet.model.pointnet import PointNetCls, PointNetSeg
from Segmentation_3d.PointNet.model.pointnet_plus import PointNetPlusCls, PointNetPlusSeg

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
    
    model_list = ["pointnet_cls", "pointnet_seg", "pointnet_plus_cls", "pointnet_plus_seg"]
    if model_name not in model_list:
        raise ValueError(f'unknown model {model_name}')
    
    if model_name == "pointnet_cls":
        return PointNetCls(class_num=class_num).to(device)
    elif model_name == "pointnet_seg":
        return PointNetSeg(class_num=class_num).to(device)
    
    pointnet_plus_dict = {}
    pointnet_plus_dict['n_samples_list'] = args.n_samples_list
    pointnet_plus_dict['radius_list'] = args.radius_list
    pointnet_plus_dict['n_points_per_group_list'] = args.n_points_per_group_list
    pointnet_plus_dict['mlp_out_channels_list'] = args.mlp_out_channels_list
        
    if model_name == "pointnet_plus_cls":
        return PointNetPlusCls(class_num, pointnet_plus_dict)
    else:
        n_feats = args.n_feats
        return PointNetPlusSeg(class_num, n_feats, pointnet_plus_dict)

def get_loss(args):
    loss_func = args.loss_func
    
    if loss_func == "ce":
        return nn.CrossEntropyLoss()
    elif loss_func == "focal":
        return FocalLoss()
    else:
        raise ValueError(f'Unknown loss function {loss_func}.')