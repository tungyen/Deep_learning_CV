import torch
from torch import nn
from torch.nn import functional as F
from typing import List
from torchvision import transforms

class VAE(nn.Module):
    
    def __init__(self, inputC, latent_dim, class_num, hidden_dim: List = None, img_size=64):
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.inputC = inputC
        module = []
        if hidden_dim is None:
            if img_size == 64:
                hidden_dim = [32, 64, 128, 256, 512]
                strides = [2, 2, 2, 2, 2]
                shrink = 32
            elif img_size == 32:
                hidden_dim = [32, 64, 128, 256]
                strides = [2, 2, 2, 2]
                shrink = 16
            elif img_size == 28:
                hidden_dim = [24, 24, 24, 24]
                strides = [1, 2, 2, 1]
                shrink = 4
                
        self.hidden_dim = hidden_dim
        final_size = int(img_size / shrink)
        self.final_size = final_size
        
        # Encoder part
        for i, h_dim in enumerate(hidden_dim):
            module.append(
                nn.Sequential(
                    nn.Conv2d(inputC, h_dim, kernel_size=3, stride=strides[i], padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(True)
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
        strides.reverse()
        
        for i in range(len(hidden_dim)-1):
            s = strides[i]
            module.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dim[i], hidden_dim[i+1], kernel_size=3, stride=s, padding=1, output_padding=s-1),
                    nn.BatchNorm2d(hidden_dim[i+1]),
                    nn.ReLU(True)
                )
            )
            
        self.decoder = nn.Sequential(*module)
        self.decoder_output_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim[-1], hidden_dim[-1], kernel_size=3, stride=strides[-1], padding=1, output_padding=strides[-1]-1),
            nn.BatchNorm2d(hidden_dim[-1]),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim[-1], out_channels=self.inputC, kernel_size=3, padding = 1),
            nn.Tanh()
        )
        
        self.class_num = class_num
        if class_num is not None:
            self.label_embedding = nn.Embedding(class_num, latent_dim)
        
    def encode(self, x: torch.Tensor):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return [mu, log_var]
    
    def decode(self, x: torch.Tensor, label):
        if self.class_num is not None and label is not None:
            x += self.label_embedding(label)
        x = self.decoder_input_layer(x)
        x = x.view(-1, self.hidden_dim[0], self.final_size, self.final_size)
        x = self.decoder(x)
        x = self.decoder_output_layer(x)
        x = torch.flatten(x, start_dim=1)
        return x
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, x: torch.Tensor, label):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z, label), mu, log_var]