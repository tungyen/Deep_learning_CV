import torch
from torch import nn
import torch.nn.functional as F
from itertools import accumulate
import operator
from timm.models.layers import DropPath, to_2tuple
import math

from core.utils.model_utils import *

class DepthwiseSeperateConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        bias=True
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=in_channels,
                stride=stride,
                bias=bias
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        )

    def forward(self, x, height, width):
        bs, n, c = x.shape
        x = x.view(bs, height, width, c).permute(0, 3, 1, 2).contiguous()
        x = self.layers(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(bs, n, -1)
        return x

class Mlp(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GeLU,
        drop=0.
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.dwconv = DepthwiseSeperateConv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x, height, width):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        n_heads=8,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        reduction_ratio=1
    ):
        super().__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.out_layer = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.reduction_ratio = reduction_ratio
        if reduction_ratio > 1:
            self.reduction = nn.Conv2d(embed_dim, embed_dim, kernel_size=reduction_ratio, stride=reduction_ratio)
            self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, height, width):
        bs, n, c = x.shape
        assert n == height * width, "Input token length does not match given height and width."
        q = self.q(x).view(bs, n, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.reduction_ratio > 1:
            x_ = x.permute(0, 2, 1).contiguous().view(bs, c, height, width)
            x_ = self.reduction(x_).view(bs, c, -1).permute(0, 2, 1).contiguous()
            x_ = self.norm(x_)
            kv = self.kv(x_).view(bs, -1, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).view(bs, -1, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(bs, n, c)
        x = self.out_layer(x)
        x = self.proj_drop(x)
        return x

class SegFormerAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        n_heads,
        mlp_ratio=4,
        qkv_bias=False,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer=nn.GeLU,
        norm_layer=nn.LayerNorm,
        reduction_ratio=1
    ):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.attn = EfficientSelfAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            reduction_ratio=reduction_ratio
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(in_channels=embed_dim, hidden_channels=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, height, width):
        x = x + self.drop_path(self.attn(self.norm1(x), height, width))
        x = x + self.drop_path(self.mlp(self.norm2(x), height, width))
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels=3,
        img_size=224,
        patch_size=7,
        stride=4,
        embed_dim=768
    ):
        super().__init__()
        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        if isinstance(patch_size, int):
            patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.height = img_size[0] // patch_size[0]
        self.width = img_size[1] // patch_size[1]
        self.num_patches = self.height * self.width
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        bs, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)
        return x, h, w

class MiT(nn.Module):
    def __init__(
        self,
        img_size=224,
        in_channels=3,
        embed_dims=[64, 128, 256, 512],
        n_heads=[2, 4, 8, 16],
        mlp_ratios=[4, 4, 4, 4],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        img_size_ratio=[1, 4, 8, 16],
        depths=[3, 4, 6, 3],
        reduction_ratios=[8, 4, 2, 1],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.depths = depths
        self.embed_dims = embed_dims

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur_idx = 0
        layer_nums = len(depths)

        self.attn_blocks = nn.ModuleList()
        self.patch_embeddings = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        for i in range(layer_nums):
            patch_embedding = OverlapPatchEmbed(
                in_channels=in_channels,
                img_size=img_size // img_size_ratio[i],
                patch_size=patch_sizes[i],
                stride=strides[i],
                embed_dim=embed_dims[i]
            )
            self.patch_embeddings.append(patch_embedding)

            attn_blocks = nn.ModuleList()
            for j in range(depths[i]):
                attn_blocks.append(
                    SegFormerAttentionBlock(
                        embed_dim=embed_dims[i],
                        n_heads=n_heads[i],
                        mlp_ratio=mlp_ratios[i],
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[cur_idx + j],
                        norm_layer=norm_layer,
                        reduction_ratio=reduction_ratios[i]
                    )
                )
            self.attn_blocks.append(attn_blocks)
            self.norm_layers.append(norm_layer(embed_dims[i]))
            cur_idx += depths[i]
            in_channels = embed_dims[i]

    def forward(self, x):
        bs = x.shape[0]
        out_feats = []

        for i in range(len(self.depths)):
            x, height, width = self.patch_embeddings[i](x)
            for blk in self.attn_blocks[i]:
                x = blk(x, height, width)
            x = self.norm_layers[i](x)
            x = x.view(bs, height, width, -1).permute(0, 3, 1, 2).contiguous()
            out_feats.append(x)
        return out_feats

class Linear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        return x

class SegFormerHead(n.Module):
    def __init__(
        self,
        in_channels=[64, 128, 256, 512],
        strides=[4, 2, 2, 2],
        embed_dim=256,
        class_num=20,
        drop_rate=0.1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.class_num = class_num
        assert len(in_channels) == len(strides)
        assert min(strides) == strides[0],
        self.strides = strides
        self.linear_layers = nn.ModuleList()

        for in_channel in in_channels:
            self.linear_layers.append(
                Linear(in_channel, embed_dim)
            )
        self.dropout = nn.Dropout(drop_rate)
        self.fusion_block = nn.Sequential(
            nn.Conv2d(len(in_channels) * embed_dim, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Conv2d(embed_dim, class_num, kernel_size=1)

    def forward(self, x):
        upsampled_feats = []
        bs, _, h, w = x[0].shape

        for i in range(len(x)):
            feat = x[i]
            feat = self.linear_layers[i](feat).permute(0, 2, 1).contiguous().view(bs, -1, feat.shape[2], feat.shape[3])
            if i != 0:
                feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
            upsampled_feats.append(feat)

        x = torch.cat(upsampled_feats, dim=1)
        x = self.fusion_block(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

class SegFormer(nn.Module):
    def __init__(
        self,
        img_size=224,
        in_channels=3,
        embed_dims=[64, 128, 256, 512],
        n_heads=[2, 4, 8, 16],
        mlp_ratios=[4, 4, 4, 4],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        img_size_ratio=[1, 4, 8, 16],
        depths=[3, 4, 6, 3],
        reduction_ratios=[8, 4, 2, 1],
        qkv_bias=True,
        drop_path_rate=0.1,
        head_embed_dim=256,
        class_num=20,
        weight_init=None
    ):
        super().__init__()
        self.backbone = MiT(
            img_size=img_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            n_heads=n_heads,
            mlp_ratios=mlp_ratios,
            patch_sizes=patch_sizes,
            strides=strides,
            img_size_ratio=img_size_ratio,
            depths=depths,
            reduction_ratios=reduction_ratios,
            qkv_bias=qkv_bias,
            drop_path_rate=drop_path_rate
        )

        self.head = SegFormerHead(
            in_channels=embed_dims,
            strides=strides,
            embed_dim=head_embed_dim,
            class_num=class_num
        )

        if weight_init is not None:
            self.apply(initialize_weights(weight_init))

    def forward(self, x):
        feats = self.backbone(x)
        out = self.head(feats)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out

    