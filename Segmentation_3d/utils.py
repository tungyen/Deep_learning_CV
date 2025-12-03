import torch
import torch.nn as nn
import os
import yaml
import torch.distributed as dist

from Segmentation_3d.PointNet.model.pointnet import PointNetCls, PointNetSemseg, PointNetPartseg
from Segmentation_3d.PointNet.model.pointnet_plus import PointNetPlusCls, PointNetPlusSemseg, PointNetPlusPartseg

def get_model(args) -> nn.Module:
    model_name = args.model
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
        args.n_points = 8192
        args.n_feats = 6
        args.task = 'semseg'
        args.train_batch_size = 32
        args.eval_batch_size = 16
        args.test_batch_size = 4
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