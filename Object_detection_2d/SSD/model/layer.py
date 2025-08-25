import torch
import torch.nn as nn
import torch.nn.init as init

class L2Norm(nn.Module):
    def __init__(self, in_channels, scale):
        super(L2Norm, self).__init__()
        self.in_channels = in_channels
        self.gamma = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.zeros((self.in_channels, )))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out