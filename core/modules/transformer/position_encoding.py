import torch
from torch import nn

def get_xpos(num_patch, start_idx=0):
    num_patch_ = int(num_patch ** 0.5)
    x_positions = torch.arange(start_idx, num_patch_ + start_idx)
    x_positions = x_positions.unsqueeze(0)
    x_positions = torch.repeat_interleave(x_positions, num_patch_, 0)
    x_positions = x_positions.reshape(-1)

    return x_positions

def get_ypos(num_patch, start_idx=0):
    num_patch_ = int(num_patch ** 0.5)
    y_positions = torch.arange(start_idx, num_patch_+start_idx)
    y_positions = torch.repeat_interleave(y_positions, num_patch_, 0)

    return y_positions


class SinusoidalPositionEmbedding2D(nn.Module):
    def __init__(self, num_patch, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim // 2
        
        x_pos = get_xpos(num_patch).reshape(-1, 1)
        x_pos_emb = self.gen_emb(x_pos)
        
        y_pos = get_ypos(num_patch).reshape(-1, 1)
        y_pos_emb = self.gen_emb(y_pos)
        
        self.pos_emb = torch.cat((x_pos_emb, y_pos_emb), -1)
        
    def gen_emb(self, pos):
        denom = torch.pow(10000, torch.arange(0, self.embed_dim, 2) / self.embed_dim)
        
        pos_emb = torch.zeros(1, pos.shape[0], self.embed_dim)
        denom = pos / denom
        pos_emb[:, :, ::2] = torch.sin(denom)
        pos_emb[:, :, 1::2] = torch.cos(denom)
        return pos_emb


class LearnedPositionEmbedding2D(nn.Module):
    def __init__(self, row_size, col_size, embed_dim):
        super().__init__()
        self.row_embedding = nn.Embedding(row_size, embed_dim // 2)
        self.col_embedding = nn.Embedding(col_size, embed_dim // 2)
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.row_embedding.weight, std=0.02)
        nn.init.uniform_(self.col_embedding.weight, std=0.02)

    def forward(self, x):
        b, c, h, w = x.shape
        row_pos = torch.arange(h, device=x.device)
        col_pos = torch.arange(w, device=x.device)

        x_emb = self.col_embedding(col_pos)
        y_emb = self.row_embedding(row_pos)
        pos_emb = torch.cat([
            x_emb.unsqueeze(0).unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(0).unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(b, 1, 1, 1)
        return pos_emb


class RelativePositionEmbedding2D(nn.Module):
    def __init__(self, embed_dim, seq_len, max_relative_dist):
        super().__init__()
        
        self.max_relative_dist = max_relative_dist
        
        self.x_embedding = nn.Embedding(max_relative_dist*2+2, embed_dim // 2)
        self.y_embedding = nn.Embedding(max_relative_dist*2+2, embed_dim // 2)
        
        x_pos = get_xpos(seq_len-1)
        x_dis = self.generate_relative_dis(x_pos)
        
        y_pos = get_ypos(seq_len-1)
        y_dis = self.generate_relative_dis(y_pos)

        self.register_buffer("x_dis", x_dis)
        self.register_buffer("y_dis", y_dis)
        
    def generate_relative_dis(self, pos):
        dis = pos.unsqueeze(0) - pos.unsqueeze(1)
        dis = torch.clamp(dis, -self.max_relative_dist, self.max_relative_dist)
        dis = dis + self.max_relative_dist + 1
        dis = F.pad(input=dis, pad=(1, 0, 1, 0), mode='constant', value=0)
        return dis
    
    def forward(self):
        x_pos_emb = self.x_embedding(self.x_dis)
        y_pos_emb = self.y_embedding(self.y_dis)
        pos_emb = torch.cat((x_pos_emb, y_pos_emb), -1)
        return pos_emb


class RotatoryPositionEmbedding2D(nn.Module):
    def __init__(self, seq_len, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim // 2
        n_patch = seq_len - 1
        
        x_pos = get_xpos(n_patch, start_idx=1).reshape(-1, 1)
        x_sin, x_cos = self.generate_rope1D(x_pos)
        
        y_pos = get_ypos(n_patch, start_idx=1).reshape(-1, 1)
        y_sin, y_cos = self.generate_rope1D(y_pos)

        self.register_buffer("x_sin", x_sin)
        self.register_buffer("x_cos", x_cos)
        self.register_buffer("y_sin", y_sin)
        self.register_buffer("y_cos", y_cos)

        
    def generate_rope1D(self, pos):
        pos = F.pad(pos, (0, 0, 1, 0))
        theta = -2 * torch.arange(start=1, end=self.embed_dim//2+1) / self.embed_dim
        theta = torch.repeat_interleave(theta, 2, 0)
        theta = torch.pow(10000, theta)
        
        val = pos * theta
        cos_val = torch.cos(val).unsqueeze(0).unsqueeze(0)
        sin_val = torch.sin(val).unsqueeze(0).unsqueeze(0)
        return sin_val, cos_val
    
    
    def forward(self, x):
        x_axis = x[:, :, :, :self.embed_dim]
        y_axis = x[:, :, :, self.embed_dim:]
        
        x_axis_cos = x_axis * self.x_cos
        x_axis_shift = torch.stack((-x_axis[:, :, :, 1::2], x_axis[:, :, :, ::2]), -1)
        x_axis_shift = x_axis_shift.reshape(x_axis.shape)
        x_axis_sin = x_axis_shift * self.x_sin
        x_axis = x_axis_cos + x_axis_sin
        
        y_axis_cos = y_axis * self.y_cos
        y_axis_shift = torch.stack((-y_axis[:, :, :, 1::2], y_axis[:, :, :, ::2]), -1)
        y_axis_shift = y_axis_shift.reshape(y_axis.shape)
        y_axis_sin = y_axis_shift * self.y_sin
        y_axis = y_axis_cos + y_axis_sin
        
        return torch.cat((x_axis, y_axis), -1)