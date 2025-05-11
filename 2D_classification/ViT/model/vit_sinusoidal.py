import torch
import torch.nn as nn
from einops import repeat
from utils import *

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
    def __init__(self, imgSize=224, patchSize=16, inputC=3, emb_dim=768):
        super(PatchEmbedding, self).__init__()
        self.imgSize = imgSize
        self.patchSize = patchSize
        self.inputC = inputC
        self.emb_dim = emb_dim
        

        self.embedding = nn.Conv2d(inputC, emb_dim, kernel_size=patchSize, stride=patchSize)
        self.patchLength = self.imgSize // self.patchSize
        self.CLS = nn.Parameter(torch.randn(1, 1, self.emb_dim))
        
        self.seq_length = self.patchLength ** 2
        self.posEmbedding = SinusoidalPositionEmbedding2D(self.seq_length, emb_dim).pos_emb
        nn.init.trunc_normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.embedding.bias, 0)
        nn.init.trunc_normal_(self.CLS, std=0.02)
        
    def forward(self, x):
        batchSize = x.shape[0]
        device = x.device
        x = self.embedding(x).flatten(2).transpose(1, 2)
        self.posEmbedding = self.posEmbedding.to(device)
        x += self.posEmbedding
        clsToken = repeat(self.CLS, 'h w c -> b (h w) c', b=batchSize)
        x = torch.cat([clsToken, x], dim=1)
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
        
        
class ViT_sinusoidal(nn.Sequential):
    def __init__(self, inputC=3, patchSize=16, emb_dim=768, imgSize=224, L=12, class_num=5, **kwargs):
        super(ViT_sinusoidal, self).__init__(
            PatchEmbedding(imgSize, patchSize, inputC, emb_dim),
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