import torch
from torch import nn

class Mlp(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=4, in_channels=3, embed_dims=96, patch_norm=True):
        super().__init__()
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=embed_dims, kernel_size=patch_size, stride=patch_size)
        if patch_norm:
            self.norm = nn.LayerNorm(embed_dims)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2)

        return x

class PatchMerging(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.LayerNorm(4 * in_channels)
        self.reduction = nn.Conv2d(in_channels=4*in_channels, out_channels=2*in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x_lu = x[:, :, 0::2, 0::2]
        x_lb = x[:, :, 1::2, 0::2]
        x_ru = x[:, :, 0::2, 1::2]
        x_rb = x[:, :, 1::2, 1::2]

        x = torch.cat([x_lu, x_lb, x_ru, x_rb], dim=1)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.reduction(x)
        return x