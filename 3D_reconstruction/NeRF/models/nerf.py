import torch
from torch import nn

# This is a class that embeds x to [x, sin(2^1 x), cos(2^1 x), ....]
class Embedding(nn.Module):
    def __init__(self, inputC, n_freq, logScale=True):
        # Input:
        #     inputC - The input channel of input data
        #     n_freq - The embedding length for both sin and cos function
        #     logScale - Decide if the embedding apply log scale
        super(Embedding, self).__init__()
        self.inputC = inputC
        self.n_freq = n_freq
        self.funcs = [torch.sin, torch.cos]
        self.outputC = inputC * (len(self.funcs)*n_freq+1)
        
        if logScale:
            self.freq_band = 2**torch.linspace(0, n_freq-1, n_freq)
        else:
            self.freq_band = torch.linspace(1, 2**(n_freq-1), n_freq)
            
    def forward(self, x):
        # Input:
        #     x - The input data with shape (B, self.inputC)
        # Output:
        #     out - The output result with shape (B, self.outputC)
        out = [x]
        for f in self.freq_band:
            for func in self.funcs:
                out += [func(f*x)]
        
        return torch.cat(out, 1)
    
class NeRF(nn.Module):
    def __init__(self, D=8, H=256, input_xyz_channels=63, input_dir_channels=27, skips=[4]):
        # Input:
        #     D - Number of layer for density encoder
        #     H - Number of hidden units in each layer
        #     input_xyz_channels - Number of input channels for 3D position
        #     input_dir_channels - Number of input channels for direction
        #     skips - Add skip connection in the D-th layer
        super(NeRF, self).__init__()
        self.D = D
        self.H = H
        self.input_xyz_channels = input_xyz_channels
        self.input_dir_channels = input_dir_channels
        self.skips = skips
        
        # Encoding for xyz
        for i in range(D):
            if i == 0:
                layer = nn.Linear(input_xyz_channels, H)
            elif i in skips:
                layer = nn.Linear(H+input_xyz_channels, H)
            else:
                layer = nn.Linear(H, H)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(H, H)
        
        # Encoding for direction
        self.direction_encoding = nn.Sequential(nn.Linear(H+input_dir_channels, H//2), nn.ReLU(True))
        
        # Output Layer
        self.sigma = nn.Linear(H, 1)
        self.rgb = nn.Sequential(nn.Linear(H//2, 3), nn.Sigmoid())
        
    def forward(self, x, sigmaOnly=False):
        # Input:
        #     x - The embedded vector of position and direction with shape (B, self.input_xyz_channels+self.input_dir_channels)
        #     sigmaOnly - Decide if only infer on sigma, if true, then x is in shape of (B, self.input_xyz_channels)
        # Output:
        #     if sigmaOnly:
        #       sigma - Output density in shape of (B, 1)
        #     else:
        #       out - Output density and rgb color with shape of (B, 4)
        if not sigmaOnly:
            input_xyz, input_dir = torch.split(x, [self.input_xyz_channels, self.input_dir_channels], dim=-1)
        else:
            input_xyz = x
            
        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)
            
        sigma = self.sigma(xyz_)
        if sigmaOnly:
            return sigma
        
        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        
        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.direction_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)
        
        out = torch.cat([rgb, sigma], -1)
        return out
        
                