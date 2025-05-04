import torch.autograd
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
import os

class Generator(nn.Module):
    def __init__(self, channels, latent_dim, class_num, input_size):
        super(Generator, self).__init__()
        dims = [32, 64, 128]
        if input_size == 64:
            dims.append(256)
            
        dims.reverse()
            
        self.latent_input_layer = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, dims[0], kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(True)
        )
        
        modules = []
        for i in range(len(dims)-1):
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(dims[i], dims[i+1], kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dims[i+1]),
                nn.ReLU(True)
            ))
            
        self.layers = nn.Sequential(*modules)
        self.fc = nn.Sequential(
            nn.ConvTranspose2d(dims[-1], channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        
        if class_num is not None:
            self.label_embedding = nn.Embedding(class_num, dims[0])
    
            
    def forward(self, z, label):
        z = self.latent_input_layer(z)
        if label is not None:
            h, w = z.shape[2], z.shape[3]
            label_emb = self.label_embedding(label).unsqueeze(2).unsqueeze(3).expand(-1, -1, h, w)
            z += label_emb
        img = self.layers(z)
        img = self.fc(img)
        return img


class Discriminator(nn.Module):
    def __init__(self, channels, class_num, input_size):
        super(Discriminator, self).__init__()
        
        self.channels = channels
        self.class_num = class_num
        self.input_size = input_size
        
        dims = [32, 64, 128]
        if input_size == 64:
            dims.append(256)
        
        
        self.input_layer = nn.Sequential(
            nn.Conv2d(channels, dims[0], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        )
        
        modules = []
        for i in range(len(dims)-1):
            modules.append(nn.Sequential(
                nn.Conv2d(dims[i], dims[i+1], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(dims[i+1]),
                nn.LeakyReLU(0.2, True),
            ))
            
        self.layers = nn.Sequential(*modules)
        self.fc = nn.Conv2d(dims[-1], 1, 4, 1, 0, bias=False)
        
        if class_num is not None:
            self.label_embedding = nn.Embedding(class_num, dims[0])
        
    def forward(self, x, label):
        x = self.input_layer(x)
        
        if label is not None:
            h, w = x.shape[2], x.shape[3]
            x += self.label_embedding(label).unsqueeze(2).unsqueeze(3).expand(-1, -1, h, w)
            
        x = self.layers(x)
        x = self.fc(x)
        return x
        