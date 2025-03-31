import torch
from torch import nn
from torch.nn import functional as F
from typing import List
from torchvision import transforms

from prior import *

def weights_init(m):
    className = m.__class__.__name__
    if className.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            pass

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
        idxes = idxes.long().view(B, H, W)
        return idxes, quantized_latents.permute(0, 3, 1, 2).contiguous(), VQ_loss
    
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
    def __init__(self, inputC, emb_num, emb_dim, n_block, class_num, prior_dim, hidden_dims: List = None, beta=0.25):
        super(VQVAE, self).__init__()
        self.emb_num = emb_num
        self.emb_dim = emb_dim
        self.beta = beta
        self.class_num = class_num
        self.inputC = inputC
        
        self.pixelcnn_prior = GatedPixelCNN(K=emb_num, inputC=emb_dim, n_block=n_block, dim=prior_dim, class_num=class_num)

        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 256]
        
        # Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(inputC, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(True)
                )
            )
            inputC = h_dim
            
        for _ in range(2):
            modules.append(ResidualBlock(inputC, inputC))
            modules.append(nn.ReLU(True))
        
        modules.append(nn.Conv2d(inputC, emb_dim, kernel_size=1, stride=1))
        self.encoder = nn.Sequential(*modules)
        self.vq_layer = VectorQuantizer(emb_num, emb_dim, beta)
        self.pixelcnn_loss_fct = nn.CrossEntropyLoss()
        
        # Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(emb_dim, hidden_dims[-1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(True)
            )
        )
        
        for _ in range(2):
            modules.append(ResidualBlock(hidden_dims[-1], hidden_dims[-1]))
        modules.append(nn.ReLU(True))
        hidden_dims.reverse()
        
        for i in range(len(hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1], kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.ReLU(True)
                )
            )
            
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1], self.inputC, kernel_size=4, stride=2, padding=1),
                nn.Tanh()
            )
        )
        self.decoder = nn.Sequential(*modules)
        self.apply(weights_init)
        
    def encode(self, x: torch.Tensor):
        x = self.encoder(x)
        return x
    
    def decode(self, z: torch.Tensor):
        x = self.decoder(z)
        x = torch.flatten(x, start_dim=1)
        return x
    
    
    def forward(self, x: torch.Tensor, label, prior_only=False):
        
        if prior_only:
            with torch.no_grad():
                encoding = self.encode(x)
                idxes, _, _ = self.vq_layer(encoding)
                idxes = idxes.detach()
            return idxes, self.pixelcnn_prior(idxes, label)
                
        encoding = self.encode(x)
        idxes, quantized_x, VQ_loss = self.vq_layer(encoding)  
        return [self.decode(quantized_x), VQ_loss]
    
    
    def prior_loss(self, x, output: torch.Tensor):
        q, logit_probs = output
        return self.pixelcnn_loss_fct(logit_probs, q)
    
    def VQVAE_loss(self, input, output):
        B = input.shape[0]
        input = input.view(B, -1)
        reconstruct, VQ_loss = output
        recons_loss = F.mse_loss(reconstruct, input)
        loss = recons_loss + VQ_loss
        return loss
    
    
    