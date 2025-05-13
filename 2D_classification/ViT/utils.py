import torch.nn as nn
from model import *

def get_model(args) -> nn.Module:
    model_name = args.model
    device = args.device
    class_num = args.class_num
    img_size = args.img_size
    patch_size = args.patch_size
    
    if model_name == "vit_sinusoidal":
        model = ViT_sinusoidal(class_num=class_num, img_size=img_size, patch_size=patch_size).to(device)
    elif model_name == "vit_relative":
        model = ViT_relative(class_num=class_num, img_size=img_size, patch_size=patch_size).to(device)
    elif model_name == "vit_rope":
        model = ViT_rope(class_num=class_num, img_size=img_size, patch_size=patch_size).to(device)
    else:
        raise ValueError(f'unknown model {model_name}')
    
    return model


    