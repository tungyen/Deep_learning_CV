import torch
import torch.nn as nn
from einops import repeat

from core.utils.model_utils import *

class SinusoidalPositionEmbedding2D(nn.Module):
    def __init__(self, n_patches, emb_dim):
        super(SinusoidalPositionEmbedding2D, self).__init__()
        self.emb_dim = emb_dim // 2
        
        x_pos = get_xpos(n_patches).reshape(-1, 1)
        x_pos_emb = self.gen_emb(x_pos)
        
        y_pos = get_ypos(n_patches).reshape(-1, 1)
        y_pos_emb = self.gen_emb(y_pos)
        
        self.pos_emb = torch.cat((x_pos_emb, y_pos_emb), -1)
        
    def gen_emb(self, pos):
        denom = torch.pow(10000, torch.arange(0, self.emb_dim, 2) / self.emb_dim)
        
        pos_emb = torch.zeros(1, pos.shape[0], self.emb_dim)
        denom = pos / denom
        pos_emb[:, :, ::2] = torch.sin(denom)
        pos_emb[:, :, 1::2] = torch.cos(denom)
        return pos_emb
        
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
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim=768, n_head=12, dropProb=0):
        super(MultiHeadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.n_head = n_head
        self.dropProb = dropProb
        self.qkv = nn.Linear(emb_dim, emb_dim*3, bias=False)
        self.dropPath = nn.Dropout(self.dropProb)
        self.proj = nn.Linear(self.emb_dim, self.emb_dim)
        self.scaling = (self.emb_dim // self.n_head) ** -0.5
        
        
    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        weight = (q @ k.transpose(-2, -1)) * self.scaling
        if mask is not None:
            fill = torch.finfo(torch.float32).min
            weight.masked_fill(~mask, fill)
        
        weight = weight.softmax(dim=-1)
        weight = self.dropPath(weight)

        
        output = (weight @ v).transpose(1, 2).reshape(B, N, C)
        output = self.proj(output)
        output = self.dropPath(output)
        return output
    
class residualAdding(nn.Module):
    def __init__(self, func):
        super(residualAdding, self).__init__()
        self.func = func
        
    def forward(self, x, **kwargs):
        ini = x
        x = self.func(x, **kwargs)
        x += ini
        return x
    
class MLP(nn.Sequential):
    def __init__(self, emb_dim=768, expansion=4, dropProb=0):
        super(MLP, self).__init__(
            nn.Linear(emb_dim, expansion * emb_dim),
            nn.GELU(),
            nn.Dropout(dropProb),
            nn.Linear(emb_dim * expansion, emb_dim),
            nn.Dropout(dropProb)
        )
        
        
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_dim=768, n_head=12, dropProb=0, expansion=4, dropProbMLP=0, **kwargs):
        super(TransformerEncoderBlock, self).__init__(
            residualAdding(nn.Sequential(
                nn.LayerNorm(emb_dim),
                MultiHeadAttention(emb_dim, n_head,**kwargs),
                nn.Dropout(dropProb)
            )),
            residualAdding(nn.Sequential(
                nn.LayerNorm(emb_dim),
                MLP(emb_dim, expansion, dropProbMLP),
                nn.Dropout(dropProb)
            )),
        )
        
        
class TransformerEncoder(nn.Sequential):
    def __init__(self, L=12, **kwargs):
        super(TransformerEncoder, self).__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(L)])


class MLP_head(nn.Module):
    def __init__(self, emb_dim=768, class_num=5):
        super(MLP_head, self).__init__()
        self.ln = nn.LayerNorm(emb_dim)
        self.fc = nn.Linear(emb_dim, class_num)
        
    def forward(self, x):
        x = x[:, 0, :]
        x = self.fc(self.ln(x))
        return x
        
        
class VitSinusoidal(nn.Sequential):
    def __init__(self, input_channel=3, patch_size=16, emb_dim=768, img_size=224, L=12, class_num=5, **kwargs):
        super(ViT_sinusoidal, self).__init__(
            PatchEmbedding(img_size, patch_size, input_channel, emb_dim),
            TransformerEncoder(L, **kwargs),
            MLP_head(emb_dim, class_num)
        )
        self.apply(_init_vit_weights)
        

def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)     