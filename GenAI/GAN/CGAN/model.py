import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

class Generator(nn.Module):
    def __init__(self, latent_dim, n_class, img_shape):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.n_class = n_class
        self.img_shape = img_shape
        img_size = img_shape[1]
        self.img_size = img_size
        C = img_shape[0]
        self.label_embedding = nn.Embedding(n_class, n_class)
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + n_class, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (img_size // 4) * (img_size // 4)),
            nn.BatchNorm1d(128 * (img_size // 4) * (img_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, C, 4, 2, 1),
            nn.Tanh(),
        )
        initialize_weights(self)
        
    def forward(self, z, label):
        emb = self.label_embedding(label)
        x = torch.cat((z, emb), dim=1)
        x = self.fc(x)
        x = x.view(-1, 128, self.img_size // 4, self.img_size // 4)
        x = self.deconv(x)
        return x
    
    
class Discriminator(nn.Module):
    def __init__(self, n_class, img_shape):
        super(Discriminator, self).__init__()
        
        self.n_class = n_class
        self.img_shape = img_shape
        C = img_shape[0]
        img_size = img_shape[1]
        self.img_size = img_size
        self.label_embedding = nn.Embedding(n_class, n_class)
        
        self.conv = nn.Sequential(
            nn.Conv2d(C + n_class, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (img_size // 4) * (img_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )
        initialize_weights(self)
        
    def forward(self, img, label):
        emb = self.label_embedding(label).unsqueeze(2).unsqueeze(3).expand(-1, -1, self.img_size, self.img_size)
        x = torch.cat((img, emb), dim=1)
        x = self.conv(x)
        B = x.shape[0]
        x = x.view(B, -1)
        x = self.fc(x)
        return x