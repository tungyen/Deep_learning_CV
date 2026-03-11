import torch
import torch.nn as nn
from einops import repeat

from core.utils.model_utils import *
from core.modules.transformer import SinusoidalPositionEmbedding2D
from core.Classification_2d.model.ViT.vit_base import VitBase
        
class VitSinusoidal(VitBase):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        weight_init=None,
        **kwargs):
        super().__init__(patch_size=patch_size, **kwargs)

        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_length = self.img_size // self.patch_size

        self.seq_length = self.patch_length ** 2
        pos_embedding = SinusoidalPositionEmbedding2D(self.seq_length, embed_dim).pos_emb
        self.register("pos_embedding", pos_embedding)

        if weight_init is not None:
            self.apply(initialize_weights(weight_init))

    def forward(self, x):
        bs = x.shape[0]
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)
        x += self.pos_embedding
        cls_token = repeat(self.CLS, 'h w c -> b (h w) c', b=bs)
        x = torch.cat([cls_token, x], dim=1)

        for blk in self.attn_blocks:
            x = blk(x)

        if not self.include_top:
            x = x[:, 1:, :]
            return x

        x = self.cls_head(x)
        return x
