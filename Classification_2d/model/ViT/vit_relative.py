import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from core.utils.model_utils import *
from core.modules.transformer import RelativePositionEmbedding2D, MultiHeadAttention
from core.Classification_2d.model.ViT.vit_base import VitBase
    
class MultiHeadAttentionRelativeDist(MultiHeadAttention):
    def __init__(self, seq_len, max_relative_dist=2, **kwargs):
        super().__init__(**kwargs)
        self.q_relative_embedding = RelativePositionEmbedding2D(self.head_dim, seq_len, max_relative_dist)
        self.v_relative_embedding = RelativePositionEmbedding2D(self.head_dim, seq_len, max_relative_dist)
        
    def forward(self, x, context=None, mask=None):
        B, N, C = x.shape

        if context is None:
            context = x

        q = self.q(x).reshape(B, N, self.num_head, self.head_dim).transpose(1, 2)
        k = self.k(context).reshape(B, -1, self.num_head, self.head_dim).transpose(1, 2)
        v = self.v(context).reshape(B, -1, self.num_head, self.head_dim).transpose(1, 2)

        weight = q @ k.transpose(-2, -1) # (B, H, N, N)
            
        q_pos = self.q_relative_embedding() # (N, N, E)
        q_pos = q_pos.permute(0, 2, 1) # (N, E, N)
        
        q = q.permute(2, 0, 1, 3) # (N, B, H, E)
        q = q.reshape(N, -1, self.head_emb_dim)
        
        relative_q = torch.matmul(q, q_pos) # (N, BH, N)
        relative_q = relative_q.reshape(N, B, -1, N)
        relative_q = relative_q.permute(1, 2, 0, 3) # (B, H, N, N)
        
        weight += relative_q
        weight *= self.scaling
        
        if mask is not None:
            fill = torch.finfo(torch.float32).min
            weight.masked_fill(~mask, fill)
            
        weight = torch.softmax(weight, dim=-1)
        weight = self.attn_drop(weight)
        x = torch.matmul(weight, v)
        
        v_pos = self.v_relative_embedding() # (N, N, E)
        weight = weight.permute(2, 0, 1, 3) # (N, B, H, N)
        weight = weight.reshape(N, -1, N) # (N, BH, N)
        relative_v = torch.matmul(weight, v_pos) # (N, BH, E)
        relative_v = relative_v.reshape(N, B, -1, self.head_emb_dim) # (N, B, H, E)
        relative_v = relative_v.permute(1, 2, 0, 3) # (B, H, N, E)
        x += relative_v
        
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
        
        
class VitRelative(VitBase):
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
            blk.attn = MultiHeadAttentionRelativeDist(
                seq_len=seq_len,
                **kwargs
            )

        if weight_init is not None:
            self.apply(initialize_weights(weight_init))

    def forward(self, x):
        bs = x.shape[0]
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)
        cls_token = repeat(self.CLS, 'h w c -> b (h w) c', b=bs)
        x = torch.cat([cls_token, x], dim=1)

        for blk in self.attn_blocks:
            x = blk(x)

        if not self.include_top:
            x = x[:, 1:, :]
            return x

        x = self.cls_head(x)
        return x
        
        
        
        