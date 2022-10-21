import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 with_bn=True,
                 with_act=True
                 ):
        super(ConvBlock, self).__init__()
        bias = not with_bn
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self.with_bn = with_bn
        if with_bn:
            self.bn = nn.BatchNorm2d(out_channels, momentum=0.1, eps=1e-5)
        self.with_act = with_act
        if with_act:
            self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        if self.with_bn:
            x = self.bn(x)
        if self.with_act:
            x = self.act(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, in_channels//2, kernel_size=1)
        self.conv2 = ConvBlock(in_channels//2, out_channels, kernel_size=3, padding=1)
        self.shortcut = shortcut

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.shortcut:
            out = out + x
        return out


class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()

