from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Conv2d(nn.Conv2d):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
    ):
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MBConv(nn.Module):
    def __init__(self, inp, oup, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = Conv2d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.GroupNorm(5 * expansion, hidden_dim),
                nn.GELU(),
                # pw-linear
                Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.GroupNorm(5, oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.GroupNorm(5 * expansion, hidden_dim),
                nn.GELU(),
                # dw
                Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.GroupNorm(5 * expansion, hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),
                # pw-linear
                Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.GroupNorm(5, oup),
            )

        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)


def conv_3x3_bn(inp, oup, downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(Conv2d(inp, oup, 3, stride, 1, bias=False), nn.GroupNorm(5, oup), nn.GELU())


class MBConvCNN(nn.Module):
    def __init__(self, in_channels, num_blocks, channels, n_layers=4):
        super().__init__()

        self.s0 = self._make_layer(conv_3x3_bn, in_channels, channels[0], num_blocks[0])
        self.s1 = self._make_layer(MBConv, channels[0], channels[1], num_blocks[1])
        self.s2 = self._make_layer(MBConv, channels[1], channels[2], num_blocks[2])
        self.s3 = self._make_layer(MBConv, channels[2], channels[3], num_blocks[3])

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.n_layers = n_layers

    def forward(self, x):
        x = self.s0(x)
        if self.n_layers == 1:
            x = self.pool(x)
            x = torch.flatten(x, 1)
            return x
        x = self.s1(x)
        if self.n_layers == 2:
            x = self.pool(x)
            x = torch.flatten(x, 1)
            return x
        x = self.s2(x)
        if self.n_layers == 3:
            x = self.pool(x)
            x = torch.flatten(x, 1)
            return x
        x = self.s3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x

    def _make_layer(self, block, inp, oup, depth):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, downsample=True))
            else:
                layers.append(block(oup, oup))
        return nn.Sequential(*layers)


def my_mbconv_vit(**kwargs: Any):
    num_blocks = [1, 1, 1, 1]
    model = MBConvCNN(num_blocks=num_blocks, **kwargs)
    return model
