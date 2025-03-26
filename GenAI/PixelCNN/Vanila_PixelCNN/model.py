import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskConv2d(nn.Module):
    def __init__(self, conv_type, *args, **kwargs):
        super(MaskConv2d, self).__init__()
        if conv_type != 'A' and conv_type != 'B':
            raise ValueError(f'unknown Convolution Type {conv_type}')
        
        self.conv = nn.Conv2d(*args, **kwargs)
        kh, kw = self.conv.weight.shape[-2:]
        mask = torch.zeros((kh, kw), dtype=torch.float32)
        mask[0: kh // 2] = 1
        mask[kh // 2, 0:kw // 2] = 1
        
        if conv_type == 'B':
            mask[kh//2, kw//2] = 1
            
        mask = mask.reshape(1, 1, kh, kw)
        self.mask = mask
        
    def forward(self, x: torch.Tensor):
        device = x.device
        self.mask = self.mask.to(device)
        self.conv.weight.data *= self.mask
        x = self.conv(x)
        return x
    
    
class ResidualBlock(nn.Module):
    def __init__(self, h_dim, batchnorm=True):
        super(ResidualBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(2*h_dim, h_dim, 1)
        self.bn1 = nn.BatchNorm2d(h_dim) if batchnorm else nn.Identity()
        self.conv2 = MaskConv2d('B', h_dim, h_dim, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(h_dim) if batchnorm else nn.Identity()
        self.conv3 = nn.Conv2d(h_dim, 2*h_dim, 1)
        self.bn3 = nn.BatchNorm2d(2*h_dim) if batchnorm else nn.Identity()
        
    def forward(self, x: torch.Tensor):
        y = self.bn1(self.conv1(self.relu(x)))
        y = self.bn2(self.conv2(self.relu(y)))
        y = self.bn3(self.conv3(self.relu(y)))
        return y + x
    
class PixelCNN(nn.Module):
    def __init__(self, inputC, n_block, h_dim, batchnorm=True, color_level=256):
        super(PixelCNN, self).__init__()
        self.inputC = inputC
        self.color_level = color_level
        self.conv1 = MaskConv2d('A', inputC, 2*h_dim, 7, 1, 3)
        self.bn1 = nn.BatchNorm2d(2*h_dim) if batchnorm else nn.Identity()
        self.residual_blocks = nn.ModuleList()
        
        for _ in range(n_block):
            self.residual_blocks.append(ResidualBlock(h_dim, batchnorm))
            self.residual_blocks.append(nn.BatchNorm2d(2*h_dim))
            nn.ReLU(True)
            
        self.relu = nn.ReLU(True)
        self.head = nn.Sequential(
            MaskConv2d('B', 2*h_dim, h_dim, 1),
            nn.ReLU(True),
            nn.Conv2d(h_dim, inputC * color_level, 1)
        )
        
    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        x = self.relu(self.bn1(self.conv1(x)))
        for block in self.residual_blocks:
            x = block(x)
            
        x = self.head(x)
        x = x.reshape(B, self.color_level, C, H, W)
        return x
        
        