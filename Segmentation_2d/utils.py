import torch.nn as nn
import torch

from Segmentation_2d.DeepLabV3.model import DeepLabV3, DeepLabV3Plus

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
    elif classname.find('Conv1d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)

def get_model(args):
    model_name = args.model
    class_num = args.class_num
    backbone = args.backbone
    momemtum = args.bn_momentum
    
    if model_name == "deeplabv3":
        model = DeepLabV3(class_num=class_num, in_channel=2048, backbone=backbone)
        set_bn_momentum(model.backbone, momentum=momemtum)
        model.classifier.apply(weights_init_xavier)
        return model
    elif model_name == "deeplabv3plus":
        model = DeepLabV3Plus(class_num=class_num, in_channel=2048, backbone=backbone)
        model.classifier.apply(weights_init_xavier)
        set_bn_momentum(model.backbone, momentum=momemtum)
        return model
    else:
        raise ValueError(f'Unknown model {model_name}')

def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum
            
def setup_args_with_dataset(dataset_type, args):
    if dataset_type == 'cityscapes':
        args.class_num = 19
        args.ignore_idx = 19
        args.train_batch_size = 16
        args.eval_batch_size = 4
        args.test_batch_size = 4
    elif dataset_type == 'voc':
        args.class_num = 21
        args.ignore_idx = 255
        args.train_batch_size = 12
        args.eval_batch_size = 16
        args.test_batch_size = 4
    else:
        raise ValueError(f'Unknown dataset {dataset_type}.')
    return args