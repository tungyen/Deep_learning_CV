import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from model.model_utils import *


class RotatoryPositionEmbedding2D(nn.Module):
    def __init__(self, seq_len, emb_dim):
        super(RotatoryPositionEmbedding2D, self).__init__()
        self.emb_dim = emb_dim // 2
        n_patch = seq_len - 1
        
        x_pos = get_xpos(n_patch, start_idx=1).reshape(-1, 1)
        self.x_sin, self.x_cos = self.generate_rope1D(x_pos)
        
        y_pos = get_ypos(n_patch, start_idx=1).reshape(-1, 1)
        self.y_sin, self.y_cos = self.generate_rope1D(y_pos)
        
    def generate_rope1D(self, pos):
        pos = F.pad(pos, (0, 0, 1, 0))
        theta = -2 * torch.arange(start=1, end=self.emb_dim//2+1) / self.emb_dim
        theta = torch.repeat_interleave(theta, 2, 0)
        theta = torch.pow(10000, theta)
        
        val = pos * theta
        cos_val = torch.cos(val).unsqueeze(0).unsqueeze(0)
        sin_val = torch.sin(val).unsqueeze(0).unsqueeze(0)
        return sin_val, cos_val
    
    
    def forward(self, x):
        device = x.device
        self.x_cos = self.x_cos.to(device)
        self.x_sin = self.x_sin.to(device)
        self.y_cos = self.y_cos.to(device)
        self.y_sin = self.y_sin.to(device)
        
        x_axis = x[:, :, :, :self.emb_dim]
        y_axis = x[:, :, :, self.emb_dim:]
        
        x_axis_cos = x_axis * self.x_cos
        x_axis_shift = torch.stack((-x_axis[:, :, :, 1::2], x_axis[:, :, :, ::2]), -1)
        x_axis_shift = x_axis_shift.reshape(x_axis.shape)
        x_axis_sin = x_axis_shift * self.x_sin
        x_axis = x_axis_cos + x_axis_sin
        
        y_axis_cos = y_axis * self.y_cos
        y_axis_shift = torch.stack((-y_axis[:, :, :, 1::2], y_axis[:, :, :, ::2]), -1)
        y_axis_shift = y_axis_shift.reshape(y_axis.shape)
        y_axis_sin = y_axis_shift * self.y_sin
        y_axis = y_axis_cos + y_axis_sin
        
        return torch.cat((x_axis, y_axis), -1)
    
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, input_channel=3, emb_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_channel = input_channel
        self.emb_dim = emb_dim
        

        self.embedding = nn.Conv2d(input_channel, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.patchLength = self.img_size // self.patch_size
        self.CLS = nn.Parameter(torch.randn(1, 1, self.emb_dim))
        
        self.seq_length = self.patchLength ** 2
        nn.init.trunc_normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.embedding.bias, 0)
        nn.init.trunc_normal_(self.CLS, std=0.02)
        
    def forward(self, x):
        batchSize = x.shape[0]
        x = self.embedding(x).flatten(2).transpose(1, 2)
        clsToken = repeat(self.CLS, 'h w c -> b (h w) c', b=batchSize)
        x = torch.cat([clsToken, x], dim=1)
        return x
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, n_head, seq_len):
        super(MultiHeadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.n_head = n_head
        self.head_emb_dim = emb_dim // n_head
        
        self.qkv = nn.Linear(emb_dim, emb_dim*3, bias=False)
        self.projection_out = nn.Linear(self.emb_dim, self.emb_dim)
        self.scaling = (self.emb_dim // self.n_head) ** -0.5
        
        self.rotary_embedding = RotatoryPositionEmbedding2D(seq_len, self.head_emb_dim)
        
    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, H, N, E)
        
        q = self.rotary_embedding(q)
        k = self.rotary_embedding(k)
        
        weight = (q @ k.transpose(-2, -1)) * self.scaling
        if mask is not None:
            fill = torch.finfo(torch.float32).min
            weight.masked_fill(~mask, fill)
        
        weight = torch.softmax(weight, dim=-1)
        x = torch.matmul(weight, v)
        
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, N, C)
        return self.projection_out(x)
    
    
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
    def __init__(self, seq_len, emb_dim=768, n_head=12, dropProb=0, expansion=4, dropProbMLP=0):
        super(TransformerEncoderBlock, self).__init__(
            residualAdding(nn.Sequential(
                nn.LayerNorm(emb_dim),
                MultiHeadAttention(emb_dim, n_head, seq_len),
                nn.Dropout(dropProb)
            )),
            residualAdding(nn.Sequential(
                nn.LayerNorm(emb_dim),
                MLP(emb_dim, expansion, dropProbMLP),
                nn.Dropout(dropProb)
            )),
        )
        
        
class TransformerEncoder(nn.Sequential):
    def __init__(self, seq_len, L=12):
        super(TransformerEncoder, self).__init__(*[TransformerEncoderBlock(seq_len) for _ in range(L)])



class MLP_head(nn.Module):
    def __init__(self, emb_dim=768, class_num=5):
        super(MLP_head, self).__init__()
        self.ln = nn.LayerNorm(emb_dim)
        self.fc = nn.Linear(emb_dim, class_num)
        
    def forward(self, x):
        x = x[:, 0, :]
        x = self.fc(self.ln(x))
        return x
        
        
class ViT_rope(nn.Sequential):
    def __init__(self, input_channel=3, patch_size=16, emb_dim=768, img_size=224, L=12, class_num=5):
        seq_len = (img_size // patch_size)**2+1
        super(ViT_rope, self).__init__(
            PatchEmbedding(img_size, patch_size, input_channel, emb_dim),
            TransformerEncoder(seq_len=seq_len, L=L),
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