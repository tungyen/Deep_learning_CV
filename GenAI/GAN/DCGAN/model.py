import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, channels, class_num):
        super(Generator, self).__init__()
        inputC = 256 if class_num is not None else 512
        self.latent_input_layer = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, inputC, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(inputC)
        )
        
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(True),
            
            nn.ConvTranspose2d(128, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
        if class_num is not None:
            self.label_embedding = nn.Embedding(class_num, 256)
        
    def forward(self, x, label):
        x = self.latent_input_layer(x)
        if label is not None:
            h, w = x.shape[2], x.shape[3]
            label_emb = self.label_embedding(label).unsqueeze(2).unsqueeze(3).expand(-1, -1, h, w)
            x = torch.cat([x, label_emb], dim=1)
        x = self.layers(x)
        return x
    
    
class Discriminator(nn.Module):
    def __init__(self, channels, class_num):
        super(Discriminator, self).__init__()
        input_c = 64 if class_num is not None else 128
        self.input_layer = nn.Sequential(
            nn.Conv2d(channels, input_c, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.layers = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )
        
        self.output_layer = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )
        
        if class_num is not None:
            self.label_embedding = nn.Embedding(class_num, 64)
            
    def forward(self, x, label):
        x = self.input_layer(x)
        if label is not None:
            h, w = x.shape[2], x.shape[3]
            label_emb = self.label_embedding(label).unsqueeze(2).unsqueeze(3).expand(-1, -1, h, w)
            x = torch.cat([x, label_emb], dim=1)
        
        x = self.layers(x)
        res = self.output_layer(x)
        return res