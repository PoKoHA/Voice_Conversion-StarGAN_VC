import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.act = nn.ReLU(inplace=True)

        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm_1 = nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm_2 = nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True)

    def forward(self, x):
        residual = x

        out = self.act(self.norm_1(self.conv_1(x)))
        out = self.norm_2(self.conv_2(out))
        out += residual
        out = self.act(out)
        return out
