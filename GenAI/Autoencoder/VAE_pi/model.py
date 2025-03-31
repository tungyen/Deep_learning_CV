from torch import nn
from torch.nn import functional as F
import torch
from typing import List

class VAE_PI(nn.Module):
    def __init__(self, inputC, latent_dim, hidden_dim: List = None):
        super(VAE_PI, self).__init__()
        self.latent_dim = latent_dim
        modules = []
        if hidden_dim == None:
            hidden_dim = [32, 64, 128, 256, 512]
        modules.append(nn.Sequential(
            nn.Linear(inputC, hidden_dim[0]),
            nn.BatchNorm1d(hidden_dim[0]),
            nn.LeakyReLU(inplace=True)
        ))
        
        # Encoder part
        for i in range(len(hidden_dim)-1):
            cur_dim = hidden_dim[i]
            next_dim = hidden_dim[i+1]
            modules.append(
                nn.Sequential(
                    nn.Linear(cur_dim, next_dim),
                    nn.BatchNorm1d(next_dim),
                    nn.LeakyReLU(inplace=True)
                )
            )
            
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dim[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dim[-1], latent_dim)
        
        
        # Decoder part
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dim[-1])
        hidden_dim.reverse()
        self.hidden_dim = hidden_dim
        
        for i in range(len(hidden_dim)-1):
            cur_dim = hidden_dim[i]
            next_dim = hidden_dim[i+1]
            modules.append(
                nn.Sequential(
                    nn.Linear(cur_dim, next_dim),
                    nn.BatchNorm1d(next_dim),
                    nn.LeakyReLU(inplace=True)
                )
            )
            
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dim[-1], inputC),
            nn.Sigmoid()
        )
        
        
    def encode(self, x: torch.Tensor):
        # x -> (B, 5)
        # mu, log_var -> (B, latent_dim)
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        log_var = self.fc_var(encoded)
        
        return [mu, log_var]
    
    def decode(self, x: torch.Tensor):
        # x -> (B, latent_dim)
        # reconstruct -> (B, 5)
        res = self.decoder_input(x)
        res = self.decoder(res)
        res = self.final_layer(res)
        return res
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, x: torch.Tensor):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), x, mu, log_var]