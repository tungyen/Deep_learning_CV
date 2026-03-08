import torch
import torch.nn as nn
from einops import repeat

from core.modules.transformer.block import Mlp

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        num_head=12,
        proj_drop_rate=0.,
        attn_drop_rate=0.,
        qkv_bias=True,
        qk_scale=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_head = num_head
        assert embed_dim % num_head == 0, "Embedding dimension must be divisible by number of heads."
        self.head_dim = embed_dim // num_head
        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.scaling = qk_scale or (self.head_dim) ** -0.5
        
    def forward(self, x, context=None, mask=None):
        B, N, C = x.shape

        if context is None:
            context = x

        q = self.q(x).reshape(B, N, self.num_head, self.head_dim).transpose(1, 2)
        k = self.k(context).reshape(B, -1, self.num_head, self.head_dim).transpose(1, 2)
        v = self.v(context).reshape(B, -1, self.num_head, self.head_dim).transpose(1, 2)

        weight = (q @ k.transpose(-2, -1)) * self.scaling
        if mask is not None:
            fill = torch.finfo(torch.float32).min
            weight.masked_fill(~mask, fill)
        
        weight = weight.softmax(dim=-1)
        weight = self.attn_drop(weight)
        
        output = (weight @ v).transpose(1, 2).reshape(B, N, C)
        output = self.proj(output)
        output = self.proj_drop(output)
        return output

class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        num_head=12,
        drop_rate=0,
        mlp_ratio=4,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        **kwargs):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)

        self.attn = MultiHeadAttention(embed_dim=embed_dim, num_head=num_head, proj_drop_rate=drop_rate, **kwargs)
        self.dropout1 = nn.Dropout(drop_rate)
        self.dropout2 = nn.Dropout(drop_rate)
        mlp_hidden_dim = int(mlp_ratio * embed_dim)
        self.mlp = MLP(in_chans=embed_dim, hidden_chans=mlp_hidden_dim, act_layer=act_layer, drop_rate=drop_rate)

    def forward(self, x):
        x = x + self.dropout1(self.attn(self.norm1(x)))
        x = x + self.dropout2(self.mlp(self.norm2(x)))
        return x