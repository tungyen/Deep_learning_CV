import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from timm.models.layers import trunc_normal_

from core.modules.transformer import TransformerEncoderBlock
from core.utils.model_utils import *
from Segmentation_2d.model.Segmenter.vit import VisionTransformer

class MaskTransformer(nn.Module):
    def __init__(
        self,
        class_num,
        patch_size,
        in_channels,
        n_layer,
        n_head,
        out_channels,
        mlp_ratio=4,
        drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=nn.LayerNorm,
        **kwargs
    ):
        super().__init__()
        self.class_num = class_num
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.n_layer = n_layer
        self.n_head = n_head
        self.out_channels = out_channels
        self.mlp_ratio = mlp_ratio
        self.scale = out_channels ** -0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layer)]
        self.block = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=out_channels,
                num_head=n_head,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                norm_layer=norm_layer,
                **kwargs
            )
        ] for i in range(n_layer))

        self.cls_embed = nn.Parameter(torch.zeros(1, class_num, out_channels))
        self.proj = nn.Linear(in_channels, out_channels)

        self.proj_patches = nn.Parameter(self.scale * torch.randn(out_channels, out_channels))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(out_channels, out_channels))

        self.norm1 = norm_layer(out_channels)
        self.norm2 = norm_layer(out_channels)
        trunc_normal_(self.cls_embed, std=0.02)

    def forward(self, x, H, W):
        x = self.proj(x)
        cls_embed = self.cls_embed.expand(x.shape[0], -1, -1)
        x = torch.cat((x, cls_embed), dim=1)

        for blk in self.block:
            x = blk(x)

        x = self.norm1(x)

        patches, cls_seg_feats = x[:, :-self.class_num], x[:, -self.class_num:]
        patches = patches @ self.proj_patches
        cls_seg_feats = cls_seg_feats @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feats = cls_seg_feats / cls_seg_feats.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feats.transpose(1, 2)
        masks = self.norm2(masks)
        masks = masks.transpose(1, 2).reshape(x.shape[0], self.class_num, H, W)

        return masks

class Segmenter(nn.Module):
    def __init__(
        self,
        img_size=(224, 224),
        class_num=20,
        in_channels=3,
        encoder_embed_dims=512,
        n_head=8,
        mlp_ratio=4,
        patch_size=8,
        n_layer=12,
        qkv_bias=True,
        drop_path_rate=0.1,
        weight_init=None,
        pretrained_weights=None,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.encoder_embed_dims = encoder_embed_dims
        self.n_head = n_head
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.qkv_bias = qkv_bias
        self.drop_path_rate = drop_path_rate
        self.class_num = class_num

        self.encoder = VisionTransformer(
            image_size=img_size,
            patch_size=patch_size,
            n_layers=n_layer,
            d_model=encoder_embed_dims,
            d_ff=encoder_embed_dims * mlp_ratio,
            n_heads=n_head,
            n_cls=class_num,
            drop_path_rate=drop_path_rate,
        )

        self.decoder = MaskTransformer(
            class_num=class_num,
            patch_size=patch_size,
            in_channels=encoder_embed_dims,
            n_layer=n_layer,
            n_head=n_head,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            **kwargs
        )

        if weight_init is not None:
            self.apply(initialize_weights(weight_init))

        if pretrained_weights is not None:
            self.encoder.load_state_dict(torch.load(pretrained_weights))

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.encoder(x, return_features=True)
        x = x[:, 1:, :]
        masks = self.decoder(x, H // self.patch_size, W // self.patch_size)

        masks = F.interpolate(masks, size=(H, W), mode="bilinear", align_corners=True)
        return masks
