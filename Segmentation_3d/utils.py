import torch
import torch.nn as nn
import os
import yaml

from Segmentation_3d.PointNet.model.pointnet import PointNetCls, PointNetSemseg, PointNetPartseg
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
    cls_class_num = args.cls_class_num
    seg_class_num = args.seg_class_num
    n_feats = args.n_feats
    
    model = None
    if model_name == "pointnet_cls":
        model = PointNetCls(cls_class_num, n_feats).to(device)
    elif model_name == "pointnet_semseg":
        model = PointNetSemseg(cls_class_num, n_feats).to(device)
    elif model_name == "pointnet_partseg":
        model = PointNetPartseg(seg_class_num, cls_class_num, n_feats).to(device)
    
    root = os.path.dirname(os.path.abspath(__file__))
    if model_name[:18] + model_name[-3:] + ".yaml" in os.listdir(os.path.join(root, "config")):
        with open(os.path.join(root, "config", model_name[:18] + model_name[-3:] + ".yaml")) as f:
            config = yaml.safe_load(f)
        
        if model_name[-3:] == "cls":
            model = PointNetPlusCls(cls_class_num, config).to(device)
        elif model_name[-6:] == "semseg":
            n_feats = args.n_feats
            model = PointNetPlusSeg(seg_class_num, n_feats, config).to(device)

    assert model is not None, f"Unknown model {model_name}."
    return model

def get_loss(args):
    loss_func = args.loss_func
    
    if loss_func == "ce":
        return nn.CrossEntropyLoss()
    elif loss_func == "focal":
        return FocalLoss()
    else:
        raise ValueError(f'Unknown loss function {loss_func}.')