import torch
import torch.nn as nn
import torch.nn.functional as F

class EMA:
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.step = 0
        
    def update_model_average(self, EMA_model: nn.Module, cur_model: nn.Module):
        for cur_param, EMA_param in zip(cur_model.parameters(), EMA_model.parameters()):
            prev, cur = EMA_param.data, cur_param.data
            EMA_param.data = self.update_average(prev, cur)
            
    def update_average(self, prev, cur):
        if prev is None:
            return cur
        return prev * self.alpha + (1-self.alpha) * cur
    
    def EMA_step(self, EMA_model, cur_model, EMA_start_step=2000):
        if self.step < EMA_start_step:
            self.reset(EMA_model, cur_model)
            self.step += 1
            return
        self.update_model_average(EMA_model, cur_model)
        self.step += 1
        
    def reset(self, EMA_model: nn.Module, cur_model: nn.Module):
        EMA_model.load_state_dict(cur_model.state_dict())
        
        
class SelfAttention(nn.Module):
    def __init__(self, channels, size, head=4):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.head = head
        self.mha = nn.MultiheadAttention(channels, head, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
        
    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        V, _ = self.mha(x_ln, x_ln, x_ln)
        V += x
        V = self.ff_self(V) + V
        return V.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)
    
    
class DoubleConv(nn.Module):
    def __init__(self, inputC, outputC, intermediateC=None, res=False):
        super().__init__()
        self.res = res
        self.inputC = inputC
        self.outputC = outputC
        if intermediateC == None:
            midC = outputC
        else:
            midC = intermediateC
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(inputC, midC, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, midC),
            nn.GELU(),
            nn.Conv2d(midC, outputC, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, outputC)
        )
        
    def forward(self, x):
        if self.res:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)
        
class Down(nn.Module):
    def __init__(self, inputC, outputC, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(inputC, inputC, res=True),
            DoubleConv(inputC, outputC)
        )
        
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, outputC)
        )
        
    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1]) # To copy size of x
        return x + emb
    
class Up(nn.Module):
    def __init__(self, inputC, outputC, emb_dim=256):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(inputC, inputC, res=True),
            DoubleConv(inputC, outputC, inputC // 2)
        )
        
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, outputC)
        )
        
    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return emb + x
    
class Unet(nn.Module):
    def __init__(self, inputC=3, outputC=3, time_emb_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_emb_dim = time_emb_dim
        self.inc = DoubleConv(inputC, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)
        
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)
        
        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, outputC, kernel_size=1)
        
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        pos_enc_sin = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_cos = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        return torch.cat([pos_enc_sin, pos_enc_cos], dim=-1)
    
    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_emb_dim)
        
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        
        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        return self.outc(x)
    

class Conditional_Unet(nn.Module):
    def __init__(self, inputC=3, outputC=3, time_emb_dim=256, classNum=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_emb_dim = time_emb_dim
        
        self.inc = DoubleConv(inputC, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)
        
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)
        
        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, outputC, kernel_size=1)
        
        if classNum is not None:
            self.label_emb = nn.Embedding(classNum, time_emb_dim)
            
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        pos_enc_sin = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_cos = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        return torch.cat([pos_enc_sin, pos_enc_cos], dim=-1)
    
    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_emb_dim)
        
        if y is not None:
            t += self.label_emb(y)
        
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        
        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        return self.outc(x)
    
if __name__ == '__main__':
    # net = UNet(device="cpu")
    net = Conditional_Unet(classNum=10, device="cpu")
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(3, 3, 64, 64)
    t = x.new_tensor([500] * x.shape[0]).long()
    y = x.new_tensor([1] * x.shape[0]).long()
    print(net(x, t, y).shape)