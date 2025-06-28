import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet1DModel(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, base_channels=64):
        super(UNet1DModel, self).__init__()

        # 编码器（下采样）
        self.enc1 = self._conv_block(input_channels, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._conv_block(base_channels * 4, base_channels * 8)

        # 瓶颈层
        self.bottleneck = self._conv_block(base_channels * 8, base_channels * 16)

        # 解码器（上采样）
        self.upconv4 = self._upconv_block(base_channels * 16, base_channels * 8)
        self.upconv3 = self._upconv_block(base_channels * 8, base_channels * 4)
        self.upconv2 = self._upconv_block(base_channels * 4, base_channels * 2)
        self.upconv1 = self._upconv_block(base_channels * 2, base_channels)

        # 输出层
        self.final_conv = nn.Conv1d(base_channels, output_channels, kernel_size=1)

    def _conv_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """卷积块：卷积 -> 激活 -> 批量归一化"""
        block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels)
        )
        return block

    def _upconv_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """上采样卷积块：转置卷积 -> 激活 -> 批量归一化"""
        block = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels)
        )
        return block

    def forward(self, x):
        # 编码器部分
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool1d(enc1, kernel_size=2))
        enc3 = self.enc3(F.max_pool1d(enc2, kernel_size=2))
        enc4 = self.enc4(F.max_pool1d(enc3, kernel_size=2))

        # 瓶颈层
        bottleneck = self.bottleneck(F.max_pool1d(enc4, kernel_size=2))

        # 解码器部分
        up4 = self.upconv4(bottleneck)
        up4 = up4 + enc4  # 跳跃连接

        up3 = self.upconv3(up4)
        up3 = up3 + enc3  # 跳跃连接

        up2 = self.upconv2(up3)
        up2 = up2 + enc2  # 跳跃连接

        up1 = self.upconv1(up2)
        up1 = up1 + enc1  # 跳跃连接

        # 输出层
        output = self.final_conv(up1)

        return output
