from torch import nn
import torch
from torch.nn import functional as F

class convBlock(nn.Module):
    def __init__(self, inputC, outputC):
        super(convBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inputC, outputC, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(outputC),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(outputC, outputC, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(outputC),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)
    
class downSampling(nn.Module):
    def __init__(self, inputC):
        super(downSampling, self).__init__()
        self.pool = nn.Sequential(
            nn.Conv2d(inputC, inputC, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(inputC),
            nn.LeakyReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.pool(x)
    
class upSampling(nn.Module):
    def __init__(self, inputC):
        super(upSampling, self).__init__()
        self.conv = nn.Conv2d(inputC, inputC // 2, 1, 1)
        
    def forward(self, x, prevFeat):
        up = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.conv(up)
        return torch.cat((prevFeat, out), dim=1)


class UNET(nn.Module):
    def __init__(self, numClasses = 21):
        super(UNET, self).__init__()
        
        # Downsampling
        self.c1 = convBlock(3, 64)
        self.pool1 = downSampling(64)
        self.c2 = convBlock(64, 128)
        self.pool2 = downSampling(128)
        self.c3 = convBlock(128, 256)
        self.pool3 = downSampling(256)
        self.c4 = convBlock(256, 512)
        self.pool4 = downSampling(512)
        self.c5 = convBlock(512, 1024)
        
        # Upsampling
        self.u1 = upSampling(1024)
        self.c6 = convBlock(1024, 512)
        self.u2 = upSampling(512)
        self.c7 = convBlock(512, 256)
        self.u3 = upSampling(256)
        self.c8 = convBlock(256, 128)
        self.u4 = upSampling(128)
        self.c9 = convBlock(128, 64)
        self.out = nn.Conv2d(64, numClasses, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(numClasses)
        
        
    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(self.pool1(x1))
        x3 = self.c3(self.pool2(x2))
        x4 = self.c4(self.pool3(x3))
        x5 = self.c5(self.pool4(x4))
        x6 = self.c6(self.u1(x5, x4))
        x7 = self.c7(self.u2(x6, x3))
        x8 = self.c8(self.u3(x7, x2))
        x9 = self.c9(self.u4(x8, x1))
        out = self.softmax(self.out(x9))
        return out
    
if __name__ == '__main__':
    x = torch.randn(2, 3, 256, 256)
    net = UNET()
    print(net(x).shape)