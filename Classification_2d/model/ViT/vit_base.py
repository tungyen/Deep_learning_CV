import torch
import torch.nn as nn
from einops import repeat

from core.utils.model_utils import *
from core.modules.transformer import PatchEmbedding, TransformerEncoderBlock 

class ClsHead(nn.Module):
    def __init__(self, embed_dim=768, class_num=5, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(embed_dim)
        self.fc = nn.Linear(embed_dim, class_num)
        
    def forward(self, x):
        x = x[:, 0, :]
        x = self.fc(self.norm(x))
        return x
        
class VitBase(nn.Module):
    def __init__(
        self,
        in_chans=3,
        embed_dim=768,
        num_heads=8,
        mlp_ratio=4.0,
        depth=12,
        patch_size=16,
        norm_layer=nn.LayerNorm,
        class_num=5,
        **kwargs
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_chans=in_chans, embed_dim=embed_dim, patch_size=patch_size)
        self.attn_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                **kwargs
            )
        ] for _ in range(depth))
        self.mlp_head = ClsHead(embed_dim=embed_dim, class_num=class_num, norm_layer=norm_layer)
            PatchEmbedding(img_size, patch_size, input_channel, emb_dim),
            TransformerEncoder(L, **kwargs),

    def forward(self, x):
        raise NotImplementedError("This function is not implemented yet")