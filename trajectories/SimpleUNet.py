# Licence: MIT
# @misc{cifar10-diffusion-2025,
#     title={CIFAR-10 Diffusion Model: Fast Training Implementation},
#     author={Karthik},
#     year={2025},
#     publisher={Hugging Face},
#     howpublished={\url{https://huggingface.co/karthik-2905/DiffusionPretrained}}
# }

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x, time_emb):
        h = self.block1(x)
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        h = self.block2(h)
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.group_norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        
        q = q.reshape(B, C, H*W).permute(0, 2, 1)
        k = k.reshape(B, C, H*W)
        v = v.reshape(B, C, H*W).permute(0, 2, 1)
        
        attn = torch.bmm(q, k) * (int(C) ** (-0.5))
        attn = F.softmax(attn, dim=2)
        
        h = torch.bmm(attn, v)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        h = self.proj_out(h)
        
        return x + h

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_emb_dim=128):
        super().__init__()
        
        # Time embedding
        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim * 4),
        )
        
        # Encoder
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.res1 = ResidualBlock(64, 64, time_emb_dim * 4)
        self.down1 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # 32->16
        
        self.res2 = ResidualBlock(64, 128, time_emb_dim * 4)
        self.down2 = nn.Conv2d(128, 128, 3, stride=2, padding=1)  # 16->8
        
        self.res3 = ResidualBlock(128, 256, time_emb_dim * 4)
        self.down3 = nn.Conv2d(256, 256, 3, stride=2, padding=1)  # 8->4
        
        # Middle
        self.mid1 = ResidualBlock(256, 512, time_emb_dim * 4)
        self.mid_attn = AttentionBlock(512)
        self.mid2 = ResidualBlock(512, 512, time_emb_dim * 4)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)  # 4->8
        self.res_up3 = ResidualBlock(256 + 256, 256, time_emb_dim * 4)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)  # 8->16
        self.res_up2 = ResidualBlock(128 + 128, 128, time_emb_dim * 4)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)  # 16->32
        self.res_up1 = ResidualBlock(64 + 64, 64, time_emb_dim * 4)
        
        # Output
        self.output = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, out_channels, 3, padding=1),
        )
    
    def forward(self, x, time):
        # Time embedding
        time_emb = self.time_embedding(time)
        time_emb = self.time_mlp(time_emb)
        
        # Encoder
        x1 = self.conv1(x)
        x1 = self.res1(x1, time_emb)
        
        x2 = self.down1(x1)
        x2 = self.res2(x2, time_emb)
        
        x3 = self.down2(x2)
        x3 = self.res3(x3, time_emb)
        
        x4 = self.down3(x3)
        
        # Middle
        x4 = self.mid1(x4, time_emb)
        x4 = self.mid_attn(x4)
        x4 = self.mid2(x4, time_emb)
        
        # Decoder
        x = self.up3(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.res_up3(x, time_emb)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.res_up2(x, time_emb)
        
        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.res_up1(x, time_emb)
        
        return self.output(x)