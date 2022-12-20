import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):

    def __init__(self, input_shape=(36, 256), n_speakers=10):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=4, padding=1, stride=2),
                                   nn.LeakyReLU(0.01, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2),
                                   nn.LeakyReLU(0.01, inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=4, padding=1, stride=2),
                                   nn.LeakyReLU(0.01, inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=4, padding=1, stride=2),
                                   nn.LeakyReLU(0.01, inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=4, padding=1, stride=2),
                                   nn.LeakyReLU(0.01, inplace=True))

        kernel_size_0 = int(input_shape[0] / np.power(2, 5))  # 1
        kernel_size_1 = int(input_shape[1] / np.power(2, 5))  # 8

        self.conv_dis = nn.Conv2d(1024, 1, kernel_size=(kernel_size_0, kernel_size_1),
                                  stride=1, padding=0, bias=False)
        self.conv_clf_spks = nn.Conv2d(1024, n_speakers, kernel_size=(kernel_size_0, kernel_size_1),
                                       stride=1, padding=0, bias=False)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        out_src = self.conv_dis(conv5)  # (B, 1, 1, 1)
        out_cls_spks = self.conv_clf_spks(conv5)  # (B, n_speakers=10, 1, 1)

        return out_src, out_cls_spks.view(out_cls_spks.size(0), out_cls_spks.size(1))


if __name__ == '__main__':
    x = torch.randn(4, 1, 36, 256).cuda()
    D = Discriminator(input_shape=(36, 256), n_speakers=10).cuda()

    out1, out2 = D(x)
    print("output: ", out1.size(), out2.size())
