import torch
import torch.nn as nn
import os
import yaml


from Segmentation_3d.PointNet.model.pointnet import PointNetCls, PointNetSemseg, PointNetPartseg
from Segmentation_3d.PointNet.model.pointnet_plus import PointNetPlusCls, PointNetPlusSemseg, PointNetPlusPartseg

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Conv1d') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Conv1d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


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
            model = PointNetSemseg(seg_class_num, n_feats)
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
                model = PointNetPlusSemseg(seg_class_num, n_feats, config)
            elif task == "partseg":
                n_feats = args.n_feats
                model = PointNetPlusPartseg(seg_class_num, cls_class_num, n_feats, config)
    else:
        raise ValueError(f'Unknown model {model_name}.')
    model.apply(weights_init_xavier)
    model = model.to(device)
    return model
    
def setup_args_with_dataset(dataset_type, args):
    if dataset_type == 'chair':
        args.cls_class_num = 4
        args.seg_class_num = 4
        args.n_points = 1600
        args.n_feats = 0
        args.task = 'semseg'
        args.train_batch_size = 64
        args.eval_batch_size = 16
        args.test_batch_size = 6
    elif dataset_type == 'modelnet40':
        args.cls_class_num = 40
        args.seg_class_num = 40
        args.n_points = 2048
        args.n_feats = 0
        args.task = 'cls'
        args.train_batch_size = 32
        args.eval_batch_size = 16
        args.test_batch_size = 6
    elif dataset_type == 's3dis':
        args.cls_class_num = 14
        args.seg_class_num = 14
        args.n_points = 4096
        args.n_feats = 6
        args.task = 'semseg'
        args.train_batch_size = 32
        args.eval_batch_size = 16
        args.test_batch_size = 1
    elif dataset_type == "shapenet":
        args.cls_class_num = 16
        args.seg_class_num = 50
        args.n_points = 2048
        args.task = 'partseg'
        if args.normal_channel:
            args.n_feats = 3
        else:
            args.n_feats = 0
        args.train_batch_size = 32
        args.eval_batch_size = 16
        args.test_batch_size = 6
    else:
        raise ValueError(f'Unknown dataset {dataset_type}.')
    return args