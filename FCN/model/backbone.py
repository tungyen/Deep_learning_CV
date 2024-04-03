import torch
import torch.nn as nn

class residualBlock(nn.Module):
    
    expansion = 4
    def __init__(self, inputC, outputC, stride=1, downsample=None, groups=1, baseWidth=64, dilation=1):
        super(residualBlock, self).__init__()
        width = int(outputC * (baseWidth/64.)) * groups
        
        self.conv1 = nn.Conv2d(inputC, width, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, 
                               padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, outputC*self.expansion, kernel_size=1, stride=stride, bias=False)
        self.bn3 = nn.BatchNorm2d(outputC * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        initial = x
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        if self.downsample != None:
            initial = self.downsample(x)
            
        x += initial
        return self.relu(x)