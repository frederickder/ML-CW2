"""
WideResNet-28-2 for CIFAR-10 semi-supervised learning.

Standard architecture used by FlexMatch (Zhang et al., 2021) and other
semi-supervised methods. This is a well-known architecture that predates
Hacohen et al. (2022).

Config from paper Appendix F.2.3:
  - WideResNet-28, widen_factor=2, leaky_slope=0.1, no dropout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, leaky_slope=0.1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                      stride=stride, bias=False)

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        return out + shortcut


class WideResNetGroup(nn.Module):
    def __init__(self, n_blocks, in_planes, out_planes, stride, leaky_slope=0.1):
        super().__init__()
        layers = [BasicBlock(in_planes, out_planes, stride, leaky_slope)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_planes, out_planes, 1, leaky_slope))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """WideResNet-28-w for CIFAR-10."""

    def __init__(self, depth=28, widen_factor=2, num_classes=10, leaky_slope=0.1):
        super().__init__()
        assert (depth - 4) % 6 == 0, "Depth must be 6n+4"
        n_blocks = (depth - 4) // 6  # 4 for WRN-28

        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.group1 = WideResNetGroup(n_blocks, channels[0], channels[1], 1, leaky_slope)
        self.group2 = WideResNetGroup(n_blocks, channels[1], channels[2], 2, leaky_slope)
        self.group3 = WideResNetGroup(n_blocks, channels[2], channels[3], 2, leaky_slope)

        self.bn = nn.BatchNorm2d(channels[3])
        self.relu = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        self.fc = nn.Linear(channels[3], num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.group1(out)
        out = self.group2(out)
        out = self.group3(out)
        out = self.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)
        return self.fc(out)
