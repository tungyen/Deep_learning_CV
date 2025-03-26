import torch
from torch import nn
from torch.nn import functional as F
from typing import List
from torchvision import transforms

class VAE(nn.Module):
    
    def __init__(self, inputC, latent_dim, hidden_dim: List = None, img_size=64):
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        module = []
        if hidden_dim is None:
            hidden_dim = [32, 64, 128, 256, 512]
        self.hidden_dim = hidden_dim
        
        final_size = int(img_size / (2 ** len(hidden_dim)))
        self.final_size = final_size
        
        # Encoder part
        for h_dim in hidden_dim:
            module.append(
                nn.Sequential(
                    nn.Conv2d(inputC, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            inputC = h_dim

        self.encoder = nn.Sequential(*module)
        self.fc_mu = nn.Linear(hidden_dim[-1] * (final_size ** 2), latent_dim)
        self.fc_var = nn.Linear(hidden_dim[-1] * (final_size ** 2), latent_dim)
        
        # Decoder part
        module = []
        self.decoder_input_layer = nn.Linear(latent_dim, hidden_dim[-1] * (final_size ** 2))
        hidden_dim.reverse()
        
        for i in range(len(hidden_dim)-1):
            module.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dim[i], hidden_dim[i+1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dim[i+1]),
                    nn.LeakyReLU()
                )
            )
            
        self.decoder = nn.Sequential(*module)
        self.decoder_output_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim[-1], hidden_dim[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dim[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim[-1], out_channels=3, kernel_size=3, padding = 1),
            nn.Sigmoid()
        )
        
        self.celeb_T = transforms.Compose([
            transforms.Resize(img_size, antialias=True),
            transforms.CenterCrop(img_size)])
        
    def encode(self, x: torch.Tensor):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return [mu, log_var]
    
    def decode(self, x: torch.Tensor):
        x = self.decoder_input_layer(x)
        x = x.view(-1, self.hidden_dim[0], self.final_size, self.final_size)
        x = self.decoder(x)
        x = self.decoder_output_layer(x)
        x = self.celeb_T(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.nan_to_num(x)
        return x
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, x: torch.Tensor):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), mu, log_var]