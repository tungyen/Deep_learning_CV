import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class PatchEmbedding(nn.Module):
    def __init__(self, imgSize=224, patchSize=16, inputC=3, embeddingC=768):
        super(PatchEmbedding, self).__init__()
        self.imgSize = imgSize
        self.patchSize = patchSize
        self.inputC = inputC
        self.embeddingC = embeddingC

        self.embedding = nn.Sequential(
            Rearrange('b c (h h1) (w w1) -> b (h w) (h1 w1 c)', h1=self.patchSize, w1=self.patchSize),
            nn.Linear(self.patchSize*self.patchSize*self.inputC, self.embeddingC)
        )
        self.patchLength = self.imgSize // self.patchSize
        self.CLS = nn.Parameter(torch.randn(1, 1, self.embeddingC))
        self.posEmbedding = nn.Parameter(torch.randn(self.patchLength**2+1, self.embeddingC))
        
    def forward(self, x):
        batchSize = x.shape[0]
        x = self.embedding(x)
        clsToken = repeat(self.CLS, 'h w c -> b (h w) c', b=batchSize)
        x = torch.cat([clsToken, x], dim=1)
        x += self.posEmbedding
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embeddingC=768, headNum=8, dropProb=0):
        super(MultiHeadAttention, self).__init__()
        self.embeddingC = embeddingC
        self.headNum = headNum
        self.dropProb = dropProb
        
        self.keys = nn.Linear(self.embeddingC ,self.embeddingC)
        self.queries = nn.Linear(self.embeddingC ,self.embeddingC)
        self.values = nn.Linear(self.embeddingC ,self.embeddingC)
        
        self.dropPath = nn.Dropout(self.dropProb)
        self.proj = nn.Linear(self.embeddingC, self.embeddingC)
        self.scaling = (self.embeddingC // self.headNum) ** -0.5
        
    def forward(self, x, mask=None):
        q = rearrange(self.queries(x), 'b n (h d) -> b h n d', h=self.headNum)
        k = rearrange(self.keys(x), 'b n (h d) -> b h n d', h=self.headNum)
        v = rearrange(self.values(x), 'b n (h d) -> b h n d', h=self.headNum)
        
        weight = torch.einsum('bhqd, bhkd -> bhqk', q, k)
        
        if mask is not None:
            fill = torch.finfo(torch.float32).min
            weight.masked_fill(~mask, fill)
            
        weight = F.softmax(weight * self.scaling, dim=-1)
        weight = self.dropPath(weight)
        
        output = torch.einsum('bhqk, bhvd -> bhqd', weight, v)
        output = rearrange(output, 'b h q d -> b q (h d)')
        output = self.proj(output)
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
        )
        
        
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, embeddingC=768, dropProb=0, expansion=4, dropProbMLP=0, **kwargs):
        super(TransformerEncoderBlock, self).__init__(
            residualAdding(nn.Sequential(
                nn.LayerNorm(embeddingC),
                MultiHeadAttention(embeddingC, **kwargs),
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



class MLP_head(nn.Sequential):
    def __init__(self, embeddingC=768, classNum = 5):
        super(MLP_head, self).__init__(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(embeddingC),
            nn.Linear(embeddingC, classNum)
        )
        
        
class ViT(nn.Sequential):
    def __init__(self, inputC=3, patchSize=16, embeddingC=768, imgSize=224, L=12, classNum=5, **kwargs):
        super(ViT, self).__init__(
            PatchEmbedding(imgSize, patchSize, inputC, embeddingC),
            TransformerEncoder(L, **kwargs),
            MLP_head(embeddingC, classNum)
        )
        
        
    
if __name__ == '__main__':
    x = torch.randn(4, 3, 224, 224)
    x = ViT()(x)
    print(x.shape)