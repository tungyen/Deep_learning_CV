import torch

def get_xpos(n_patches, start_idx=0):
    n_patches_ = int(n_patches ** 0.5)
    x_positions = torch.arange(start_idx, n_patches_ + start_idx)
    x_positions = x_positions.unsqueeze(0)
    x_positions = torch.repeat_interleave(x_positions, n_patches_, 0)
    x_positions = x_positions.reshape(-1)

    return x_positions

def get_ypos(n_patches, start_idx=0):
    n_patches_ = int(n_patches ** 0.5)
    y_positions = torch.arange(start_idx, n_patches_+start_idx)
    y_positions = torch.repeat_interleave(y_positions, n_patches_, 0)

    return y_positions