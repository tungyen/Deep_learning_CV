import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class backBone(nn.Module):
    
    def __init__(self):
        super(backBone, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
    def forward(self, x):
        return self.resnet(x)
    
class RPN(nn.Module):
    def __init__(self)