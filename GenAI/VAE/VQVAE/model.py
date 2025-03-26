import torch
from torch import nn
from torch.nn import functional as F
from typing import List
from torchvision import transforms


class VectorQuantizer(nn.Module):
    def __init__(self, emb_num: int, emb_dim: int, beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.emb_num = emb_num
        self.emb_dim = emb_dim
        self.beta = beta
        
        self.emb = nn.Embedding(self.emb_num, self.emb_dim)
        self.emb.weight.data.uniform_(-1 / self.emb_num, 1 / self.emb_num)
        
    def forward(self, latents: torch.Tensor):
        latents = latents.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = latents.shape
        flatten_latents = latents.view(-1, self.emb_dim) # (BHW, emb_dim)
        
        dist = torch.sum(flatten_latents ** 2, dim=1, keepdim=True) + \
                torch.sum(self.emb.weight ** 2, dim=1) - \
                2 * torch.matmul(flatten_latents, self.emb.weight.t()) # (BHW, emb_num)
                
        idxes = torch.argmin(dist, dim=1).unsqueeze(1) # (BHW, 1)
        
        device = latents.device
        encoding_onehot = torch.zeros(idxes.shape[0], self.emb_num, device=device)
        encoding_onehot.scatter_(1, idxes, 1) # (BHW, emb_num)
        
        quantized_latents = torch.matmul(encoding_onehot, self.emb.weight) # (BHW, emb_dim)
        quantized_latents = quantized_latents.view(B, H, W, C)
        
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        emb_loss = F.mse_loss(quantized_latents, latents.detach())
        
        VQ_loss = commitment_loss * self.beta + emb_loss
        quantized_latents = latents + (quantized_latents - latents).detach()
        return quantized_latents.permute(0, 3, 1, 2).contiguous(), VQ_loss
    
class ResidualBlock(nn.Module):
    def __init__(self, inputC: int, outputC: int):
        super(ResidualBlock, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(inputC, outputC, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(outputC, outputC, kernel_size=1, bias=False)
        )
        
    def forward(self, x: torch.Tensor):
        return x + self.resblock(x)


class VQVAE(nn.Module):
    def __init__(self, inputC: int, emb_num: int, emb_dim: int, hidden_dims: List = None, beta=0.25, img_size=64):
        super(VQVAE, self).__init__()
        
        self.emb_num = emb_num
        self.emb_dim = emb_dim
        self.img_size = img_size
        self.beta = beta

        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 256]
        
        # Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(inputC, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU()
                )
            )
            inputC = h_dim
            
        for _ in range(6):
            modules.append(ResidualBlock(inputC, inputC))
        modules.append(nn.LeakyReLU())
        
        modules.append(
            nn.Sequential(
                nn.Conv2d(inputC, emb_dim, kernel_size=1, stride=1),
                nn.LeakyReLU()
            )
        )
        self.encoder = nn.Sequential(*modules)
        self.vq_layer = VectorQuantizer(emb_num, emb_dim, beta)
        
        # Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(emb_dim, hidden_dims[-1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU()
            )
        )
        
        for _ in range(6):
            modules.append(ResidualBlock(hidden_dims[-1], hidden_dims[-1]))
        modules.append(nn.LeakyReLU())
        hidden_dims.reverse()
        
        for i in range(len(hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1], kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU()
                )
            )
            
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1], 3, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid()
            )
        )
        self.decoder = nn.Sequential(*modules)
        self.celeb_T = transforms.Compose([
            transforms.Resize(img_size, antialias=True),
            transforms.CenterCrop(img_size)])
        
    def encode(self, x: torch.Tensor):
        x = self.encoder(x)
        return x
    
    def decode(self, z: torch.Tensor):
        x = self.decoder(z)
        x = self.celeb_T(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.nan_to_num(x)
        return x
    
    
    def forward(self, x: torch.Tensor):
        encoding = self.encode(x)
        print("Shape of encoding: ", encoding.shape)
        quantized_x, VQ_loss = self.vq_layer(encoding)
        print("Shape of quantized encoding: ", quantized_x.shape)
        return [self.decode(quantized_x), VQ_loss]
    
    