from torch import nn

class MaskedCNN(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        self.mask_type = mask_type
        assert mask_type in ['A', 'B'], "Unknown Mask Type"
        super(MaskedCNN, self).__init__(*args, **kwargs)
        self.mask = self.weight.data.clone()
        _, _, height, width = self.weight.size()
        self.mask.fill_(1)
        if mask_type =='A':
            self.mask[:,:,height//2,width//2:] = 0
            self.mask[:,:,height//2+1:,:] = 0
        else:
            self.mask[:,:,height//2,width//2+1:] = 0
            self.mask[:,:,height//2+1:,:] = 0
        self.mask = self.mask.to("cuda")
    def forward(self, x):
        self.weight.data*=self.mask
        return super(MaskedCNN, self).forward(x)

class PixelCNN(nn.Module):
    def __init__(self, inputC, color_level, no_layers=8, channels=64, kernel = 7, device=None):
        super(PixelCNN, self).__init__()
        self.no_layers = no_layers
        self.kernel = kernel
        self.channels = channels
        self.layers = {}
        self.device = device

        modules = []
        modules.append(nn.Sequential(
            MaskedCNN('A', inputC, channels, kernel, 1, kernel//2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True)
        ))
        
        for _ in range(no_layers):
            modules.append(nn.Sequential(
            MaskedCNN('B', channels, channels, kernel, 1, kernel//2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True)
        ))
            
        self.layers = nn.Sequential(*modules)
            
        self.out = nn.Conv2d(channels, color_level, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        for block in self.layers:
            x = block(x)
            
        out = self.out(x)
        out = out.reshape(B, -1, C, H, W)
        return out