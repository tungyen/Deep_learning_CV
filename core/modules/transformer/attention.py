import torch
import torch.nn as nn
from einops import repeat

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim=768, n_head=12, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_head = n_head
        assert emb_dim % n_head == 0, "Embedding dimension must be divisible by number of heads."
        self.head_dim = emb_dim // n_head
        self.q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.v = nn.Linear(emb_dim, emb_dim, bias=False)
        self.drop_out = nn.Dropout(drop_rate)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(self.emb_dim, self.emb_dim)
        self.scaling = (self.head_dim) ** -0.5
        
        
    def forward(self, x, context=None,mask=None):
        B, N, C = x.shape

        if context is None:
            context = x

        q = self.q(x).reshape(B, N, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k(context).reshape(B, -1, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v(context).reshape(B, -1, self.n_head, self.head_dim).transpose(1, 2)

        weight = (q @ k.transpose(-2, -1)) * self.scaling
        if mask is not None:
            fill = torch.finfo(torch.float32).min
            weight.masked_fill(~mask, fill)
        
        weight = weight.softmax(dim=-1)
        weight = self.attn_drop(weight)
        
        output = (weight @ v).transpose(1, 2).reshape(B, N, C)
        output = self.proj(output)
        output = self.drop_out(output)
        return output