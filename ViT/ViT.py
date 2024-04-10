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
        
        self.CLS = nn.Parameter(torch.randn(1, 1, self.embeddingC))
        
    def forward(self, x):
        batchSize = x.shape[0]
        print(batchSize)
        x = self.embedding(x)
        clsToken = repeat(self.CLS, 'h w c -> b (h w) c', b=batchSize)
        x = torch.cat([clsToken, x], dim=1)
        return x
    
if __name__ == '__main__':
    x = torch.randn(4, 3, 224, 224)
    PE = PatchEmbedding()
    output = PE(x)
    print(output.shape)