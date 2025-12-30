import torch
from torch import nn
from timm.models.layers import trunc_normal_

from Classification_2d.model.ViT.attention_block import *

def window_partition(x, window_size):
    """
    (bs, c, h, w) -> (bs * n_windows, c, window_size, window_size)
    """
    b, c, h ,w = x.shape
    x = x.view(b, c, h // window_size, window_size, w // window_size, window_size)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(-1, c, window_size, window_size)
    return x

def window_reverse(windows, window_size, img_size):
    b, c, _, _ = windows.shape
    h, w = img_size
    h_windows = h // window_size
    w_windows = w // window_size
    n_windows = h_windows * w_windows
    windows = windows.reshape(b // n_windows, h_windows, w_windows, c, window_size, window_size).permute(0, 3, 4, 1, 5, 2)
    x = windows.reshape(b // n_windows, c, h_windows * window_size, w_windows * window_size)
    return x

def get_relative_position_index(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij')) # (2, wh, ww)
    coords_flatten = torch.flatten(coords, 1) # (2, wh*ww)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] # (2, wh*ww, wh*ww)
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += window_size - 1
    relative_coords[:, :, 1] += window_size - 1
    relative_coords[:, :, 0] *= window_size - 1
    relative_position_index = relative_coords.sum(-1) # (wh*ww, wh*ww)
    return relative_position_index