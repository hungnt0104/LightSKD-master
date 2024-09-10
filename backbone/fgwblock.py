import torch
import torch.nn as nn
from backbone.cbam import CBAM

def conv3x3(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels, affine=True, momentum=0.99, eps=1e-3),
        nn.ReLU(inplace=True)
    )

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=padding, dilation=dilation, groups=in_channels,
                                   bias=bias)
        self.bnd = nn.BatchNorm2d(in_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, stride, 0, 1, 1, bias=bias)
        self.bnp = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.relu(self.bnd(out))
        out = self.pointwise(out)
        out = self.relu(self.bnp(out))
        return out

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, keep_dim=False):
        super(Block, self).__init__()

        self.keep_dim = keep_dim
        stride_sep_conv2 = 2
        if keep_dim:
            stride_sep_conv2 = 1
        self.conv = conv3x3(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.sep_conv1 = SeparableConv2d(in_channels, out_channels, kernel_size=3, bias=False, padding=1)
        self.sep_conv2 = SeparableConv2d(out_channels, out_channels, kernel_size=3, stride=stride_sep_conv2, bias=False, padding=1)
        self.cbam = CBAM(out_channels)
        self.maxp = nn.MaxPool2d(kernel_size=2, stride=2) if not keep_dim else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.conv(x)
        if not self.keep_dim:
            residual = self.maxp(residual)

        out = self.sep_conv1(x)
        out = self.sep_conv2(out)
        out = self.cbam(out)
        out += residual
        # out = self.relu(out)
        return out
