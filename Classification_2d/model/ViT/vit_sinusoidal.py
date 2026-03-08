import torch
import torch.nn as nn
from einops import repeat

from core.utils.model_utils import *
from core.modules.transformer import SinusoidalPositionEmbedding2D, PatchEmbedding, TransformerEncoderBlock
        
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, input_channel=3, emb_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_channel = input_channel
        self.emb_dim = emb_dim
        

        self.embedding = nn.Conv2d(input_channel, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_length = self.img_size // self.patch_size
        self.CLS = nn.Parameter(torch.randn(1, 1, self.emb_dim))
        
        self.seq_length = self.patch_length ** 2
        self.pos_embedding = SinusoidalPositionEmbedding2D(self.seq_length, emb_dim).pos_emb
        nn.init.trunc_normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.embedding.bias, 0)
        nn.init.trunc_normal_(self.CLS, std=0.02)
        
    def forward(self, x):
        batchSize = x.shape[0]
        device = x.device
        x = self.embedding(x).flatten(2).transpose(1, 2)
        self.pos_embedding = self.pos_embedding.to(device)
        x += self.pos_embedding
        cls_token = repeat(self.CLS, 'h w c -> b (h w) c', b=batchSize)
        x = torch.cat([cls_token, x], dim=1)
        return x  

class MLP_head(nn.Module):
    def __init__(self, emb_dim=768, class_num=5):
        super(MLP_head, self).__init__()
        self.ln = nn.LayerNorm(emb_dim)
        self.fc = nn.Linear(emb_dim, class_num)
        
    def forward(self, x):
        x = x[:, 0, :]
        x = self.fc(self.ln(x))
        return x
        
class VitSinusoidal(nn.Module):
    def __init__(
        self,
        input_channel=3,
        patch_size=16,
        emb_dim=768,
        img_size=224,
        L=12,
        class_num=5,
        weight_init=None,
        **kwargs):
        super(VitSinusoidal, self).__init__(
            PatchEmbedding(img_size, patch_size, input_channel, emb_dim),
            TransformerEncoder(L, **kwargs),
            MLP_head(emb_dim, class_num)
        )
        if weight_init is not None:
            self.apply(initialize_weights(weight_init))  