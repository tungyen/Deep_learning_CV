import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from Classification_2d.ViT.model.model_utils import *


class RelativePositionEmbedding2D(nn.Module):
    def __init__(self, emb_dim, seq_len, max_relative_dist):
        super(RelativePositionEmbedding2D, self).__init__()
        
        self.max_relative_dist = max_relative_dist
        
        self.x_embedding = nn.Embedding(max_relative_dist*2+2, emb_dim // 2)
        self.y_embedding = nn.Embedding(max_relative_dist*2+2, emb_dim // 2)
        
        x_pos = get_xpos(seq_len-1)
        self.x_dis = self.generate_relative_dis(x_pos)
        
        y_pos = get_ypos(seq_len-1)
        self.y_dis = self.generate_relative_dis(y_pos)
        
    def generate_relative_dis(self, pos):
        dis = pos.unsqueeze(0) - pos.unsqueeze(1)
        dis = torch.clamp(dis, -self.max_relative_dist, self.max_relative_dist)
        dis = dis + self.max_relative_dist + 1
        dis = F.pad(input=dis, pad=(1, 0, 1, 0), mode='constant', value=0)
        return dis
    
    def forward(self, device):
        self.x_dis = self.x_dis.to(device)
        self.y_dis = self.y_dis.to(device)
        x_pos_emb = self.x_embedding(self.x_dis)
        y_pos_emb = self.y_embedding(self.y_dis)
        pos_emb = torch.cat((x_pos_emb, y_pos_emb), -1)
        return pos_emb
    

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
    def __init__(self, emb_dim, n_head, seq_len, max_relative_dist):
        super(MultiHeadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.n_head = n_head
        self.head_emb_dim = emb_dim // n_head
        self.qkv = nn.Linear(emb_dim, emb_dim*3, bias=False)
        self.project_out = nn.Linear(emb_dim, emb_dim, bias=False)
        self.scaling = (self.emb_dim // self.n_head) ** -0.5
        
        self.q_relative_embedding = RelativePositionEmbedding2D(self.head_emb_dim, seq_len, max_relative_dist)
        self.v_relative_embedding = RelativePositionEmbedding2D(self.head_emb_dim, seq_len, max_relative_dist)
        
    def forward(self, x, mask=None):
        B, N, C = x.shape
        device = x.device
        qkv = self.qkv(x).reshape(B, N, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, H, N, E)
        
        weight = q @ k.transpose(-2, -1) # (B, H, N, N)
            
        q_pos = self.q_relative_embedding(device) # (N, N, E)
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
        x = torch.matmul(weight, v)
        
        v_pos = self.v_relative_embedding(device) # (N, N, E)
        weight = weight.permute(2, 0, 1, 3) # (N, B, H, N)
        weight = weight.reshape(N, -1, N) # (N, BH, N)
        relative_v = torch.matmul(weight, v_pos) # (N, BH, E)
        relative_v = relative_v.reshape(N, B, -1, self.head_emb_dim) # (N, B, H, E)
        relative_v = relative_v.permute(1, 2, 0, 3) # (B, H, N, E)
        x += relative_v
        
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, N, C)
        return self.project_out(x)
    
    
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
    def __init__(self, seq_len, emb_dim=768, n_head=12, dropProb=0, expansion=4, max_relative_dist=2, dropProbMLP=0):
        super(TransformerEncoderBlock, self).__init__(
            residualAdding(nn.Sequential(
                nn.LayerNorm(emb_dim),
                MultiHeadAttention(emb_dim, n_head, seq_len, max_relative_dist),
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
        
        
class ViT_relative(nn.Sequential):
    def __init__(self, input_channel=3, patch_size=16, emb_dim=768, img_size=224, L=12, class_num=5):
        seq_len = (img_size // patch_size)**2+1
        super(ViT_relative, self).__init__(
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
        
        
        