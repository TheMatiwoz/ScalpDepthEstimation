import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tools


class UNet(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels

        self.conv_x2 = ConvX2(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.last = OutConv(64, 1)

    def forward(self, x):
        x1 = self.conv_x2(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.last(x)
        return x


class ConvX2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_2d(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            ConvX2(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvX2(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class WarpingLayer(nn.Module):
    def __init__(self, f, B, epsilon=1.0e-8):
        super().__init__()
        self.B = B
        self.f = f
        self.epsilon = epsilon

    def forward(self, x):
        i2, d = x
        D = np.zeros(d.shape, dtype=np.uint8) + self.f * self.B
        D = np.divide(D, d + self.epsilon)
        return tools.inverse_warping.warp(i2, D)

