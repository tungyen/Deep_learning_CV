import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from core.utils.model_utils import *
from core.modules.transformer import RotatoryPositionEmbedding2D, MultiHeadAttention
from Classification_2d.model.ViT.vit_base import VitBase
    
class MultiHeadAttentionRope(MultiHeadAttention):
    def __init__(self, seq_len, **kwargs):
        super().__init__(**kwargs)
        self.rotary_embedding = RotatoryPositionEmbedding2D(seq_len, self.head_dim)
        
    def forward(self, x, context=None, mask=None):
        B, N, C = x.shape

        if context is None:
            context = x

        q = self.q(x).reshape(B, N, self.num_head, self.head_dim).transpose(1, 2)
        k = self.k(context).reshape(B, -1, self.num_head, self.head_dim).transpose(1, 2)
        v = self.v(context).reshape(B, -1, self.num_head, self.head_dim).transpose(1, 2)
        
        q = self.rotary_embedding(q)
        k = self.rotary_embedding(k)
        
        weight = (q @ k.transpose(-2, -1)) * self.scaling
        if mask is not None:
            fill = torch.finfo(torch.float32).min
            weight.masked_fill(~mask, fill)
        
        weight = torch.softmax(weight, dim=-1)
        weight = self.attn_drop(weight)
        x = torch.matmul(weight, v)
        
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
        
class VitRope(VitBase):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        weight_init=None,
        **kwargs):
        super().__init__(patch_size=patch_size, **kwargs)
        patch_len = img_size // patch_size
        seq_len = patch_len ** 2 + 1

        for blk in self.attn_blocks:
            blk.attn = MultiHeadAttentionRope(
                seq_len=seq_len,
                **kwargs
            )

        if weight_init is not None:
            self.apply(initialize_weights(weight_init))

    def forward(self, x):
        bs, c, h_ori, w_ori = x.shape
        x = self.patch_embedding(x)
        h_patch, w_patch = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        cls_token = repeat(self.CLS, 'h w c -> b (h w) c', b=bs)
        x = torch.cat([cls_token, x], dim=1)

        for blk in self.attn_blocks:
            x = blk(x)

        if not self.include_top:
            x = x[:, 1:, :]
            return x

        x = self.cls_head(x)
        return x