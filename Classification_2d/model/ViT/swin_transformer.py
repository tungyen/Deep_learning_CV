import torch
from torch import nn
from timm.models.layers import trunc_normal_

from Classification_2d.model.ViT.attention_block import *
from core.utils.model_utils import *

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

def get_mask(img_size, window_size, shift_size):
    img_mask = torch.zeros((1, 1, img_size, img_size))
    h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    cnt = 0

    for h in h_slices:
        for w in w_slices:
            img_mask[:, :, h, w] = cnt
            cnt += 1
    
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask

class WindowAttention(nn.Module):
    def __init__(self, window_size, n_heads, embed_dims, qkv_bias=True, attention_drop=0., proj_drop=0.):
        super().__init__()
        self.window_size = window_size
        self.n_heads = n_heads
        self.scale = (embed_dims // n_heads) ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_reverse - 1) * (2 * window_size - 1), n_heads)
        )

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        relative_position_index = get_relative_position_index(window_size)
        self.register_buffer('relative_position_index', relative_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3)
        self.attn_drop = nn.Dropout(attention_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        bs, c, window_size, _ = x.shape
        x = x.permute(0, 2, 3, 1)
        qkv = self.qkv(x)
        qkv = qkv.reshape(bs, window_size * window_size, c // self.n_heads, self.n_heads, 3)
        q = qkv[:, :, :, :, 0].permute(0, 3, 1, 2)
        k = qkv[:, :, :, :, 1].permute(0, 3, 1, 2)
        v = qkv[:, :, :, :, 2].permute(0, 3, 1, 2)

        q *= self.scale
        attn_weight = q @ k.transpose(-2, -1)
        relative_position_embedding = self.relative_position_bias_table[self.relative_position_index.view(1)]
        relative_position_embedding = relative_position_embedding.view(self.window_size ** 2, self.window_size ** 2, -1).permute(2, 0, 1)
        attn_weight = attn_weight + relative_position_embedding.unsqueeze(0)

        if mask is not None:
            n_windows = mask.shape[0]
            mask = mask.unsqueeze(1).unsqueeze(0)
            attn_weight = attn_weight.view(bs // n_windows, n_windows, self.n_heads, window_size ** 2, window_size ** 2)
            attn_weight = attn_weight + mask
            attn_weight = attn_weight.view(-1, self.n_heads, window_size ** 2, window_size ** 2)

        attn_weight = self.softmax(attn_weight)
        x = attn_weight @ v
        x = x.transpose(1, 2).reshape(bs, window_size, window_size, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 3, 1, 2)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dims,
        n_heads,
        img_size,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.,
        attn_drop=0.,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.n_heads = self.n_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio= mlp_ratio
        self.img_size = img_size

        self.norm_0 = norm_layer(embed_dims)
        self.attn = WindowAttention(window_size, n_heads, embed_dims, qkv_bias, attn_drop, drop)
        self.norm_1 = norm_layer(embed_dims)
        self.mlp = Mlp(in_channels=embed_dims, hidden_channels=int(mlp_ratio * embed_dims))

        if self.shift_size > 0:
            attn_mask = get_mask(img_size, window_size, shift_size)
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x):
        residual = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm_0(x)
        x = x.permute(0, 3, 1, 2)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        
        x_windows = window_partition(x, self.window_size)
        attn_windows = self.attn(x_windows, self.attn_mask)
        x = window_reverse(attn_windows, self.window_size, self.img_size)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        x = residual + x
        x = x.permute(0, 2, 3, 1)
        x = x + self.mlp(self.norm_1(x))
        x = x.permute(0, 3, 1, 2)
        return x


class SwinTransformerBlockStack(nn.Module):
    def __init__(
        self,
        embed_dims,
        n_heads,
        img_size,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.,
        attn_drop=0.,
        norm_layer=norm_layer
    ):
        super().__init__()
        self.WMHA = SwinTransformerBlock(
            embed_dims, n_heads, img_size, window_size, 0, mlp_ratio, qkv_bias, drop, attn_drop, norm_layer
        )
        self.SWMHA = SwinTransformerBlock(
            embed_dims, n_heads, img_size, window_size, window_size // 2, mlp_ratio, qkv_bias, drop, attn_drop, norm_layer
        )

    def forward(self, x):
        x = self.WMHA(x)
        x = self.SWMHA(x)
        return x


class SwinTransformer(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size=4,
        in_channels=3,
        embed_dims=96,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        patch_norm=True,
        depths=[2, 2, 6, 2],
        n_heads=[3, 6, 12, 24],
        norm_layer=nn.LayerNorm,
        include_top=True,
        weight_init=None
    ):
        super().__init__()
        self.n_layers = len(depths)
        self.embed_dims = embed_dims
        self.patch_norm = patch_norm
        self.out_channels = int(embed_dims * (2 ** (self.n_layers - 1)))
        self.mlp_ratio = mlp_ratio

        self.patch_embedding = PatchEmbedding(patch_size, in_channels, embed_dims, patch_norm)
        self.pos_drop = nn.Dropout(drop_rate)

        img_size //= patch_size
        in_channels = embed_dims
        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            cur_layer = nn.ModuleList()
            if i > 0:
                cur_layer.append(PatchMerging(in_channels=in_channels))
                img_size //= 2
                in_channels *= 2
            for _ in range(depths[i] // 2):
                cur_layer.append(
                    SwinTransformerBlockStack(
                        embed_dims=in_channels,
                        n_heads=n_heads[i],
                        img_size=img_size,
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        norm_layer=norm_layer
                    )
                )

            cur_layer = nn.Sequential(*cur_layer)
            self.layers.append(cur_layer)

        self.norm = norm_layer(self.out_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if include_top:
            self.cls_head = nn.Linear(self.out_channels, class_num)
        else:
            self.cls_head = None

        if weight_init is not None:
            self.apply(initialize_weights(weight_init))

    
    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)

        if self.cls_head is not None:
            x = self.cls_head(x)
        return x

    