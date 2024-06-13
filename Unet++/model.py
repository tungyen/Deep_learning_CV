import torch
from torch import nn as nn

class convBlock(nn.Module):
    def __init__(self, input_ch, mid_ch, out_ch):
        super(convBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_ch, mid_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x
    
    
class Unet_plus2(nn.Module):
    def __init__(self, numClass, input_ch=3, deep_supervise=False):
        super(Unet_plus2, self).__init__()
        
        ch_list = [32, 64, 128, 256, 512]
        self.deep_supervise = deep_supervise
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv00 = convBlock(input_ch, ch_list[0], ch_list[0])
        self.conv10 = convBlock(ch_list[0], ch_list[1], ch_list[1])
        self.conv20 = convBlock(ch_list[1], ch_list[2], ch_list[2])
        self.conv30 = convBlock(ch_list[2], ch_list[3], ch_list[3])
        self.conv40 = convBlock(ch_list[3], ch_list[4], ch_list[4])
        
        self.conv01 = convBlock(ch_list[0]+ch_list[1], ch_list[0], ch_list[0])
        self.conv11 = convBlock(ch_list[1]+ch_list[2], ch_list[1], ch_list[1])
        self.conv21 = convBlock(ch_list[2]+ch_list[3], ch_list[2], ch_list[2])
        self.conv31 = convBlock(ch_list[3]+ch_list[4], ch_list[3], ch_list[3])
        
        self.conv02 = convBlock(ch_list[0]*2+ch_list[1], ch_list[0], ch_list[0])
        self.conv12 = convBlock(ch_list[1]*2+ch_list[2], ch_list[1], ch_list[1])
        self.conv22 = convBlock(ch_list[2]*2+ch_list[3], ch_list[2], ch_list[2])
        
        self.conv03 = convBlock(ch_list[0]*3+ch_list[1], ch_list[0], ch_list[0])
        self.conv13 = convBlock(ch_list[1]*3+ch_list[2], ch_list[1], ch_list[1])
        
        self.conv04 = convBlock(ch_list[0]*4+ch_list[1], ch_list[0], ch_list[0])
        
        if self.deep_supervise:
            self.final1 = nn.Conv2d(ch_list[0], numClass, kernel_size=1)
            self.final2 = nn.Conv2d(ch_list[0], numClass, kernel_size=1)
            self.final3 = nn.Conv2d(ch_list[0], numClass, kernel_size=1)
            self.final4 = nn.Conv2d(ch_list[0], numClass, kernel_size=1)
        else:
            self.final = nn.Conv2d(ch_list[0], numClass, kernel_size=1)
            
    def forward(self, x):
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool(x00))
        x20 = self.conv20(self.pool(x10))
        x30 = self.conv30(self.pool(x20))
        x40 = self.conv40(self.pool(x30))
        
        x01 = self.conv01(torch.cat([x00, self.up(x10)], 1))
        x11 = self.conv11(torch.cat([x10, self.up(x20)], 1))
        x21 = self.conv21(torch.cat([x20, self.up(x30)], 1))
        x31 = self.conv31(torch.cat([x30, self.up(x40)], 1))
        
        x02 = self.conv02(torch.cat([x00, x01, self.up(x11)], 1))
        x12 = self.conv12(torch.cat([x10, x11, self.up(x21)], 1))
        x22 = self.conv22(torch.cat([x20, x21, self.up(x31)], 1))
        
        x03 = self.conv03(torch.cat([x00, x01, x02, self.up(x12)], 1))
        x13 = self.conv13(torch.cat([x10, x11, x12, self.up(x21)], 1))
        
        x04 = self.conv04(torch.cat([x00, x01, x02, x03, self.up(x13)], 1))
        
        if self.deep_supervise:
            out1 = self.final1(x01)
            out2 = self.final2(x02)
            out3 = self.final2(x03)
            out4 = self.final2(x04)
            return [out1, out2, out3, out4]
        
        else:
            out = self.final(x04)
            return out