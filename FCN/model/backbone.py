import torch
import torch.nn as nn

class residualBlock(nn.Module):
    
    expansion = 4
    def __init__(self, inputC, outputC, stride=1, downsample=None, groups=1, baseWidth=64, dilation=1):
        super(residualBlock, self).__init__()
        width = int(outputC * (baseWidth/64.)) * groups
        
        self.conv1 = nn.Conv2d(inputC, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, 
                               padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, outputC*self.expansion, kernel_size=1, bias=False)
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
    
class ResNet50(nn.Module):
    def __init__(self, block, layers, numClasses=1000, groups=1, groupWidth=64):
        super(ResNet50, self).__init__()
        self.inputC = 64
        self.dilation = 1
        self.groups = groups
        self.groupWidth = groupWidth
        self.layers = layers
        
        self.conv1 = nn.Conv2d(3, self.inputC, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inputC)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.layerGenerator(64, block, 3)
        self.layer2 = self.layerGenerator(128, block, 4, stride=2)
        self.layer3 = self.layerGenerator(256, block, 6, stride=2)
        self.layer4 = self.layerGenerator(512, block, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, numClasses)
        
    def layerGenerator(self, inputC, block, blockNumber, stride=1, dilate=False):
        
        downSample = None
        prevDilation = self.dilation
        
        if dilate == True:
            self.dilation *= stride
            stride = 1
            
        if stride != 1 or self.inputC != inputC * block.expansion:
            downSample = nn.Sequential(
                nn.Conv2d(self.inputC, inputC*block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(inputC * block.expansion)
            )
        layers = []
        layers.append(block(self.inputC, inputC, stride, downSample, self.groups, self.groupWidth, prevDilation))
        self.inputC = inputC * block.expansion
        for _ in range(1, blockNumber):
            layers.append(block(self.inputC, inputC, stride, downSample, self.groups, self.groupWidth, self.dilation))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x