import os
import sys
import torch
import torch.nn as nn
from torch.hub import download_url_to_file
from torch.hub import urlparse
from torch.hub import HASH_REGEX

from core.utils.ddp_utils import is_main_process, synchronize

# Loading the model weight.
def cache_url(url, model_dir=None, progress=True):
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv("TORCH_HOME", "~/.torch"))
        model_dir = os.getenv("TORCH_MODEL_ZOO", os.path.join(torch_home, "models"))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if filename == "model_final.pkl":
        filename = parts.path.replace("/", "_")
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file) and is_main_process():
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = HASH_REGEX.search(filename)
        if hash_prefix is not None:
            hash_prefix = hash_prefix.group(1)
            if len(hash_prefix) < 6:
                hash_prefix = None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    synchronize()
    return cached_file

def load_state_dict_from_url(url, map_location='cpu'):
    cached_file = cache_url(url)
    return torch.load(cached_file, map_location=map_location)


# Model Init
def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Conv1d') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
    elif classname.find('Conv1d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)

WEIGHT_INIT_DICT = {
    "xavier": weights_init_xavier,
    "kaiming": weights_init_kaiming
}

def initialize_weights(weight_init_name):
    if weight_init_name not in WEIGHT_INIT_DICT:
        raise ValueError(f'Unknown weight initialization method {weight_init_name}')
    return WEIGHT_INIT_DICT[weight_init_name]


# Position Embedding
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