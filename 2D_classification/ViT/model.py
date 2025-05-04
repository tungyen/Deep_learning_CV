import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

def get_xpos(n_patches, start_idx=0):
    n_patches_ = int(n_patches ** 0.5)
    x_positions = torch.arange(start_idx, n_patches_ + start_idx)
    x_positions = x_positions.unsqueeze(0)
    x_positions = torch.repeat_interleave(x_positions, n_patches_, 0)
    x_positions = x_positions.reshape(-1)

    return x_positions

def get_ypos(n_patches, start_idx=0):
    n_patches_ = int(n_patches ** 0.5)
    y_positions = torch.arange(start_idx, n_patches_+start_idx)
    y_positions = torch.repeat_interleave(y_positions, n_patches_, 0)

    return y_positions

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
    def __init__(self, imgSize=224, patchSize=16, inputC=3, embeddingC=768):
        super(PatchEmbedding, self).__init__()
        self.imgSize = imgSize
        self.patchSize = patchSize
        self.inputC = inputC
        self.embeddingC = embeddingC
        

        self.embedding = nn.Conv2d(inputC, embeddingC, kernel_size=patchSize, stride=patchSize)
        self.patchLength = self.imgSize // self.patchSize
        self.CLS = nn.Parameter(torch.randn(1, 1, self.embeddingC))
        
        self.seq_length = self.patchLength ** 2
        
        # self.posEmbedding = nn.Parameter(torch.randn(self.patchLength**2+1, self.embeddingC))
        self.posEmbedding = SinusoidalPositionEmbedding2D(self.seq_length, embeddingC).pos_emb
        # nn.init.trunc_normal_(self.posEmbedding, std=0.02)
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
    def __init__(self, embeddingC=768, headNum=12, dropProb=0):
        super(MultiHeadAttention, self).__init__()
        self.embeddingC = embeddingC
        self.headNum = headNum
        self.dropProb = dropProb
        self.qkv = nn.Linear(embeddingC, embeddingC*3, bias=False)
        self.dropPath = nn.Dropout(self.dropProb)
        self.proj = nn.Linear(self.embeddingC, self.embeddingC)
        self.scaling = (self.embeddingC // self.headNum) ** -0.5
        
        # self.query_pos_embedding = RelativeDistancePosEmbedding(embeddingC, seq_len)
        # self.value_pos_embedding = RelativeDistancePosEmbedding(embeddingC, seq_len)
        
    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.headNum, C // self.headNum).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        weight = (q @ k.transpose(-2, -1)) * self.scaling
        if mask is not None:
            fill = torch.finfo(torch.float32).min
            weight.masked_fill(~mask, fill)
        
        weight = weight.softmax(dim=-1)
        weight = self.dropPath(weight)
        
        # q_pos = self.query_pos_embedding().permute(0, 2, 1)
        
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
    def __init__(self, embeddingC=768, expansion=4, dropProb=0):
        super(MLP, self).__init__(
            nn.Linear(embeddingC, expansion * embeddingC),
            nn.GELU(),
            nn.Dropout(dropProb),
            nn.Linear(embeddingC * expansion, embeddingC),
            nn.Dropout(dropProb)
        )
        
        
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, embeddingC=768, headNum=12, dropProb=0, expansion=4, dropProbMLP=0, **kwargs):
        super(TransformerEncoderBlock, self).__init__(
            residualAdding(nn.Sequential(
                nn.LayerNorm(embeddingC),
                MultiHeadAttention(embeddingC, headNum,**kwargs),
                nn.Dropout(dropProb)
            )),
            residualAdding(nn.Sequential(
                nn.LayerNorm(embeddingC),
                MLP(embeddingC, expansion, dropProbMLP),
                nn.Dropout(dropProb)
            )),
        )
        
        
class TransformerEncoder(nn.Sequential):
    def __init__(self, L=12, **kwargs):
        super(TransformerEncoder, self).__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(L)])



class MLP_head(nn.Module):
    def __init__(self, embeddingC=768, classNum = 5):
        super(MLP_head, self).__init__()
        self.ln = nn.LayerNorm(embeddingC)
        self.fc = nn.Linear(embeddingC, classNum)
        
    def forward(self, x):
        x = x[:, 0, :]
        x = self.fc(self.ln(x))
        return x
        
        
class ViT(nn.Sequential):
    def __init__(self, inputC=3, patchSize=16, embeddingC=768, imgSize=224, L=12, classNum=5, **kwargs):
        super(ViT, self).__init__(
            PatchEmbedding(imgSize, patchSize, inputC, embeddingC),
            TransformerEncoder(L, **kwargs),
            MLP_head(embeddingC, classNum)
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
    
if __name__ == '__main__':
    x = torch.randn(4, 3, 224, 224)
    x = ViT()(x)
    print(x.shape)