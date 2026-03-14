import torch
from torch import nn

class Mlp(nn.Module):
    def __init__(self, in_chans, hidden_chans=None, out_chans=None, act_layer=nn.GELU, drop_rate=0.):
        super().__init__()
        out_chans = out_chans or in_chans
        hidden_chans = hidden_chans or in_chans
        self.fc1 = nn.Linear(in_chans, hidden_chans)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_chans, out_chans)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, patch_size=4, patch_norm=True):
        super().__init__()
        self.proj = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        if patch_norm:
            self.norm = nn.LayerNorm(embed_dim)
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
    def __init__(self, in_chans):
        super().__init__()
        self.norm = nn.LayerNorm(4 * in_chans)
        self.reduction = nn.Conv2d(in_channels=4*in_chans, out_channels=2*in_chans, kernel_size=1, bias=False)

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