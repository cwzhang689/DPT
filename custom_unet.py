import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]  # [B, half_dim]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.res_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.activation(self.norm1(h))
        h = h + self.time_mlp(t_emb).unsqueeze(-1)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.dropout(h)
        return self.activation(h + self.res_conv(x))

class UNet1DModel(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, base_channels=64, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.enc1 = ResidualBlock1D(input_channels, base_channels, time_emb_dim)
        self.enc2 = ResidualBlock1D(base_channels, base_channels * 2, time_emb_dim)
        self.enc3 = ResidualBlock1D(base_channels * 2, base_channels * 4, time_emb_dim)
        self.enc4 = ResidualBlock1D(base_channels * 4, base_channels * 8, time_emb_dim)

        self.bottleneck = ResidualBlock1D(base_channels * 8, base_channels * 16, time_emb_dim)

        self.upconv4 = ResidualBlock1D(base_channels * 16, base_channels * 8, time_emb_dim)
        self.upconv3 = ResidualBlock1D(base_channels * 8, base_channels * 4, time_emb_dim)
        self.upconv2 = ResidualBlock1D(base_channels * 4, base_channels * 2, time_emb_dim)
        self.upconv1 = ResidualBlock1D(base_channels * 2, base_channels, time_emb_dim)

        self.final_conv = nn.Conv1d(base_channels, output_channels, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        enc1 = self.enc1(x, t_emb)
        enc2 = self.enc2(F.max_pool1d(enc1, kernel_size=2), t_emb)
        enc3 = self.enc3(F.max_pool1d(enc2, kernel_size=2), t_emb)
        enc4 = self.enc4(F.max_pool1d(enc3, kernel_size=2), t_emb)

        bottleneck = self.bottleneck(F.max_pool1d(enc4, kernel_size=2), t_emb)

        up4 = F.interpolate(bottleneck, scale_factor=2, mode='nearest')
        up4 = self.upconv4(up4 + enc4, t_emb)

        up3 = F.interpolate(up4, scale_factor=2, mode='nearest')
        up3 = self.upconv3(up3 + enc3, t_emb)

        up2 = F.interpolate(up3, scale_factor=2, mode='nearest')
        up2 = self.upconv2(up2 + enc2, t_emb)

        up1 = F.interpolate(up2, scale_factor=2, mode='nearest')
        up1 = self.upconv1(up1 + enc1, t_emb)

        return self.final_conv(up1)
