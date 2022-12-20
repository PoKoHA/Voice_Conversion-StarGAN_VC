import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import ResidualBlock


class Generator(nn.Module):

    def __init__(self, n_speakers=10):
        super().__init__()

        self.proj = nn.Sequential(nn.Conv2d(1+n_speakers, 64, kernel_size=(3, 9), padding=(1, 4), bias=False),
                                  nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True))
        self.d1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(4, 8), padding=(1, 3), stride=(2, 2), bias=False),
                                nn.InstanceNorm2d(128, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True))
        self.d2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(4, 8), padding=(1, 3), stride=(2, 2), bias=False),
                                nn.InstanceNorm2d(256, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True))

        self.res1 = ResidualBlock(in_channels=256, out_channels=256)
        self.res2 = ResidualBlock(in_channels=256, out_channels=256)
        self.res3 = ResidualBlock(in_channels=256, out_channels=256)
        self.res4 = ResidualBlock(in_channels=256, out_channels=256)
        self.res5 = ResidualBlock(in_channels=256, out_channels=256)
        self.res6 = ResidualBlock(in_channels=256, out_channels=256)

        self.u1 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
                                nn.InstanceNorm2d(128, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True))
        self.u2 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
                                nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True))

        self.last = nn.Conv2d(64, 1, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x, c):
        """
        x: MCEP(mel-cepstral coefficients)
        c: One-Hot Representation
        """
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat((x, c), dim=1)

        out = self.proj(x)
        out = self.d1(out)
        out = self.d2(out)

        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)
        out = self.res6(out)

        out = self.u1(out)
        out = self.u2(out)
        out = self.last(out)

        return out


if __name__ == '__main__':
    mc = torch.randn(4, 1, 36, 256).cuda()
    c = torch.randn(4, 10).cuda()

    g = Generator().cuda()
    print("Output", g(mc, c).size())
