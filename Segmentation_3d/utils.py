import torch
import torch.nn as nn
import os
import yaml

from Segmentation_3d.PointNet.model.pointnet import PointNetCls, PointNetSemseg, PointNetPartseg, PointNetPartseg2
from Segmentation_3d.PointNet.model.pointnet_plus import PointNetPlusCls, PointNetPlusSeg

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)

def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, mat_diff_loss_scale=0.001):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.mat_diff_loss_scale = mat_diff_loss_scale
        
    def forward(self, prediction, annotation, trans_feats=None):
        loss = nn.CrossEntropyLoss()
        CE_loss = loss(prediction, annotation)
        pt = torch.exp(-CE_loss)
        focal_loss = self.alpha * (1-pt) ** self.gamma * CE_loss
        if trans_feats is not None:
            mat_diff_loss = feature_transform_reguliarzer(trans_feats)
            focal_loss += mat_diff_loss
        return focal_loss
    
class CrossEntropyLoss(nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super().__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, prediction, annotation, trans_feats=None):
        loss = nn.CrossEntropyLoss()
        CE_loss = loss(prediction, annotation)
        if trans_feats is not None:
            mat_diff_loss = feature_transform_reguliarzer(trans_feats)
            CE_loss += mat_diff_loss
        return CE_loss
    
def get_model(args) -> nn.Module:
    model_name = args.model
    device = args.device
    cls_class_num = args.cls_class_num
    seg_class_num = args.seg_class_num
    n_feats = args.n_feats
    task = args.task

    if model_name == "pointnet":
        if task == "cls":
            model = PointNetCls(cls_class_num, n_feats)
        elif task == "semseg":
            model = PointNetSemseg(cls_class_num, n_feats)
        elif task == "partseg":
            model = PointNetPartseg(seg_class_num, cls_class_num, n_feats)
        else:
            raise ValueError(f'Unknown task {task}.')
    elif model_name in ["pointnet_plus_ssg", "pointnet_plus_msg"]:
        root = os.path.dirname(os.path.abspath(__file__))
        config_name = model_name + '_' + task[-3:] + ".yaml"
        if config_name in os.listdir(os.path.join(root, "config")):
            with open(os.path.join(root, "config", config_name)) as f:
                config = yaml.safe_load(f)
            if task == "cls":
                model = PointNetPlusCls(cls_class_num, config)
            elif task == "semseg":
                n_feats = args.n_feats
                model = PointNetPlusSeg(seg_class_num, n_feats, config)
    else:
        raise ValueError(f'Unknown model {model_name}.')
    model.apply(weights_init)
    model = model.to(device)
    return model

def get_loss(args):
    loss_func = args.loss_func
    if loss_func == "ce":
        return CrossEntropyLoss()
    elif loss_func == "focal":
        return FocalLoss()
    else:
        raise ValueError(f'Unknown loss function {loss_func}.')
    
def setup_args_with_dataset(dataset_type, args):
    if dataset_type == 'chair':
        args.cls_class_num = 4
        args.seg_class_num = 4
        args.n_points = 1600
        args.n_feats = 0
        args.task = 'semseg'
    elif dataset_type == 'modelnet40':
        args.cls_class_num = 40
        args.seg_class_num = 40
        args.n_points = 2048
        args.n_feats = 0
        args.task = 'cls'
    elif dataset_type == 's3dis':
        args.cls_class_num = 14
        args.seg_class_num = 14
        args.n_points = 4096
        args.n_feats = 6
        args.task = 'semseg'
    elif dataset_type == "shapenet":
        args.cls_class_num = 16
        args.seg_class_num = 50
        args.n_points = 2048
        args.task = 'partseg'
        if args.normal_channel:
            args.n_feats = 3
        else:
            args.n_feats = 0
    else:
        raise ValueError(f'Unknown dataset {dataset_type}.')
    return args