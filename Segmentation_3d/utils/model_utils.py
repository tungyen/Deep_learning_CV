import torch.nn as nn
import torch

def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum

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
    elif classname.find('Conv1d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)

WEIGHT_INIT_DICT = {
    "xavier": weights_init_xavier,
    "kaiming": weights_init_kaiming
}

def initialize_weights(weight_init_name):
    if weight_init_name not in WEIGHT_INIT_DICT:
        raise ValueError(f'Unknown weight initialization method {weight_init_name}')
    return WEIGHT_INIT_DICT[weight_init_name]