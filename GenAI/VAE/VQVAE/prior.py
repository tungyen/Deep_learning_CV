import torch
import torch.nn as nn

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, inputC, outputC, kernel_size, padding, use_cuda=True):
        super(MaskedConv2d, self).__init__(inputC, outputC, kernel_size, padding=padding)
        
        self.mask = torch.ones((outputC, inputC, kernel_size, kernel_size)).float()
        if use_cuda:
            self.mask = self.mask.to('cuda')
        h, w = kernel_size, kernel_size
        
        if mask_type == 'A':
            self.mask[:, :, h//2, w//2:] = 0
            self.mask[:, :, h//2+1:, :] = 0
        else:
            self.mask[:, :, h//2, w//2+1:] = 0
            self.mask[:, :, h//2+1:, :] = 0
    
    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)
    
    
class LayerNorm(nn.LayerNorm):
    def __init__(self, color_condition, *args, **kwargs):
        super(LayerNorm, self).__init__(*args, **kwargs)
        self.color_condition = color_condition
        
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x_shape = x.shape
        
        if self.color_condition:
            x = x.contiguous().view(*(x_shape[:-1] + (3, -1)))
            
        x = super().forward(x)
        if self.color_condition:
            x = x.view(*x_shape)
        return x.permute(0, 3, 1, 2).contiguous()
    

class StackLayerNorm(nn.Module):
    def __init__(self, dim):
        super(StackLayerNorm, self).__init__()
        self.h_layernorm = LayerNorm(False, dim)
        self.v_layernorm = LayerNorm(False, dim)
        
    def forward(self, x: torch.Tensor):
        vx, hx = x.chunk(2, dim=1)
        vx, hx = self.v_layernorm(vx), self.h_layernorm(hx)
        return torch.cat((vx, hx), dim=1)
        
        
class GatedConv2d(nn.Module):
    def __init__(self, mask_type, inputC, outputC, k=7, padding=3, use_cuda=True):
        super(GatedConv2d, self).__init__()
        
        self.mask_type = mask_type
        self.vertical = nn.Conv2d(inputC, 2*outputC, k, padding=padding, bias=False)
        self.horizontal = nn.Conv2d(inputC, 2*outputC, (1, k), padding=(0, padding), bias=False)
        self.vh = nn.Conv2d(2*outputC, 2*outputC, kernel_size=1, bias=False)
        self.hh = nn.Conv2d(outputC, outputC, kernel_size=1, bias=False)
        
        self.v_mask = self.vertical.weight.data.clone()
        self.h_mask = self.horizontal.weight.data.clone()
        if use_cuda:
            self.v_mask = self.v_mask.to('cuda')
            self.h_mask = self.h_mask.to('cuda')
        
        self.v_mask.fill_(1)
        self.h_mask.fill_(1)
        
        self.v_mask[:, :, k//2+1:, :] = 0
        self.h_mask[:, :, :, k//2+1:] = 0
        if mask_type == 'A':
            self.h_mask[:, :, :, k//2] = 0
            
    def down_shift(self, x):
        x = x[:, :, :-1, :]
        pad = nn.ZeroPad2d((0, 0, 1, 0))
        return pad(x)
    
    def forward(self, x: torch.Tensor):
        vx, hx = x.chunk(2, dim=1)
        self.vertical.weight.data *= self.v_mask
        self.horizontal.weight.data *= self.h_mask
        
        vx = self.vertical(vx)
        hx_new = self.horizontal(hx)
        hx_new = hx_new + self.vh(self.down_shift(vx))
        
        vx1, vx2 = vx.chunk(2, dim=1)
        vx = torch.tanh(vx1) + torch.sigmoid(vx2)
        
        hx1, hx2 = hx_new.chunk(2, dim=1)
        hx_new = torch.tanh(hx1) + torch.sigmoid(hx2)
        hx_new = self.hh(hx_new)
        hx = hx + hx_new
        
        return torch.cat((vx, hx), dim=1)
    
    
class GatedPixelCNN(nn.Module):
    def __init__(self, K, inputC, n_block, dim):
        super(GatedPixelCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=K, embedding_dim=inputC)
        self.input_layer = MaskedConv2d('A', inputC, dim, 7, 3)
        
        modules = []
        for _ in range(n_block-2):
            modules.extend([nn.ReLU(True), GatedConv2d('B', dim, dim, 7, 3)])
            modules.append(StackLayerNorm(dim))
            
        self.output_layer = MaskedConv2d('B', dim, K, 7, 3)
        self.net = nn.Sequential(*modules)
        
    def forward(self, x: torch.Tensor):
        z = self.embedding(x).permute(0, 3, 1, 2).contiguous()
        
        z = self.input_layer(z)
        z = self.net(torch.cat((z, z), dim=1)).chunk(2, dim=1)[1]
        z = self.output_layer(z)
        return z
        
            
        