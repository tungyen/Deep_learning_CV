import torch.nn as nn
from model import resnet34, resnet50, resnet101, resnext50_32x4d, resnext101_32x8d

def get_model(args) -> nn.Module:
    model_name = args.model
    device = args.device
    class_num = args.class_num
    
    if model_name == "resnet34":
        model = resnet34(class_num=class_num).to(device)
    elif model_name == "resnet50":
        model = resnet50(class_num=class_num).to(device)
    elif model_name == "resnet101":
        model = resnet101(class_num=class_num).to(device)
    elif model_name == "resnext50_32x4d":
        model = resnext50_32x4d(class_num=class_num).to(device)
    elif model_name == "resnext101_32x8d":
        model = resnext101_32x8d(class_num=class_num).to(device)
    else:
        raise ValueError(f'unknown model {model_name}')

    return model


    