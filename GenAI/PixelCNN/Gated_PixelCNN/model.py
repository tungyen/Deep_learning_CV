import torch
import torch.nn as nn

class MaskConv2d(nn.Conv2d):
    def __init__(self, inputC, outputC, kernel_size, mask_type, gate_type, use_gpu=True):
        if mask_type != 'A' and mask_type != 'B':
            raise ValueError(f'unknown Convolution Type {mask_type}')
        
        if gate_type != 'H' and gate_type != 'V':
            raise ValueError(f'unknown Gate Type {gate_type}')
        
        if gate_type == 'H':
            pad = (0, (kernel_size-1) // 2)
            ks = (1, kernel_size)
        else:
            pad = (kernel_size-1) // 2
            ks = kernel_size
            
        super(MaskConv2d, self).__init__(inputC, outputC, kernel_size=ks, padding=pad)
        self.mask = torch.zeros_like(self.weight)
        if use_gpu:
            self.mask = self.mask.cuda()
        
        _, _, H, W = self.mask.shape
        if mask_type == 'A':
            if gate_type == 'V':
                self.mask[:, :, 0:H // 2, :] = 1
            else:
                self.mask[:, :, :, 0:W // 2] = 1
        else:
            if gate_type == 'V':
                self.mask[:, :, 0:H // 2, :] = 1
                self.mask[:, :, H // 2, :] = 1
            else:
                self.mask[:, :, :, 0:W // 2 + 1] = 1
    
    def __call__(self, x):
        self.weight.data *= self.mask
        return super().__call__(x)
    
    
class GatedConvLayer(nn.Module):
    def __init__(self, inputC, dim, mask_type, class_num, kernel_size=3):
        super(GatedConvLayer, self).__init__()
        self.dim = dim
        self.inputC = inputC
        self.mask_type = mask_type
        self.kernel_size = kernel_size
        
        self.v_conv = MaskConv2d(inputC, dim*2, kernel_size, mask_type, 'V')
        self.h_conv = MaskConv2d(inputC, dim*2, kernel_size, mask_type, 'H')
        self.vh_conv = nn.Conv2d(2*dim, 2*dim, kernel_size=1)
        self.hh_conv = nn.Conv2d(dim, dim, kernel_size=1)
        
        if class_num is not None:
            self.label_embedding = nn.Embedding(class_num, dim)
        
    def GateActivate(self, x, label):
        if label is not None:
            label_emb = self.label_embedding(label).unsqueeze(2).unsqueeze(3)
            return torch.tanh(x[:, :self.dim]+label_emb) * torch.sigmoid(x[:, self.dim:]+label_emb)
        else:
            return torch.tanh(x[:, :self.dim]) * torch.sigmoid(x[:, self.dim:])
    
    def forward(self, x, label=None):   
        v, h = x
        ov = self.v_conv(v)
        oh = self.h_conv(h)
        vh = self.vh_conv(ov)
        oh = vh + oh
        
        ov = self.GateActivate(ov, label)
        oh = self.GateActivate(oh, label)
        oh = self.hh_conv(oh)
        
        if self.mask_type == 'B':
            oh = oh + h
        
        return [ov, oh]
    
    
class GatedPixelCNN(nn.Module):
    def __init__(self, n_block, inputC, dim, class_num=None, color_level=16, ksA=5, ksB=3):
        super(GatedPixelCNN, self).__init__()
        modules = []
        modules.append(GatedConvLayer(inputC, dim, 'A', class_num, kernel_size=ksA))
        
        for _ in range(n_block):
            modules.append(GatedConvLayer(dim, dim, 'B', class_num, kernel_size=ksB))
            
        self.layers = nn.Sequential(*modules)
        self.out = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.ReLU(True),
            nn.Conv2d(dim, color_level, 1)
        )
        if class_num is not None:
            self.label_embedding = nn.Embedding(class_num, dim)
        
    def forward(self, x, label=None):
        B, C, H, W = x.shape
        x = [x, x]
        for layer in self.layers:
            x = layer(x, label)
        
        if label is not None:
            label_emb = self.label_embedding(label).unsqueeze(2).unsqueeze(3)
            x[1] += label_emb
        output = self.out(x[1])
        output = output.view(B, -1, C, H, W)
        return output
        
            
        