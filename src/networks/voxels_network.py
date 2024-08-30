import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th


class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


### REMARK: Change to 4
def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(dims, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D
    """

    def __init__(self, channels, out_channels=None, use_conv=False, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        out_channels = default(out_channels, channels)
        if use_conv:
            self.conv = conv_nd(dims, channels, out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class ResBlock_V(nn.Module):
    def __init__(
        self,
        channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        activation=SiLU(),
        skip_h=None,
        padding=1,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        if skip_h is not None:
            self.skip_norm = normalization(channels)

        self.in_norm = normalization(channels)
        self.act1 = activation
        self.in_conv = conv_nd(dims, channels, self.out_channels, 3, padding=padding)

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            activation,
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=padding
            )
        else:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=padding
            )

    def forward(self, x, skip_h=None):
        B, H, W, L, C = x.shape
        h = self.in_norm(x)
        if skip_h is not None:
            # print("res", x.shape, skip_h.shape)
            skip_h = self.skip_norm(skip_h)
            h = (h + skip_h) / math.sqrt(2)
        h = self.act1(h)
        h = self.in_conv(h)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class Encoder_Down_4(nn.Module):
    def __init__(self, channel_in, channel_out, dropout=0.0, activation=SiLU()):
        super().__init__()
        self.actvn = F.relu

        self.in_conv = nn.Conv3d(channel_in, 64, 3, padding=1, stride=2)
        self.res_1 = ResBlock_V(64, dropout, out_channels=128, dims=3)

        self.down_1 = Downsample(128, False, dims=3)
        self.res_2 = ResBlock_V(128, dropout, out_channels=128, dims=3)
        # self.down_2 = Downsample(128, False, dims=3)
        # self.res_3 = ResBlock_V(128,
        #                         dropout,
        #                         out_channels=128,
        #                         dims=3)
        # self.down_3 = Downsample(128, False, dims=3)

        self.out_conv = nn.Sequential(
            normalization(128), activation, conv_nd(3, 128, channel_out, 3, padding=1)
        )

    def forward(self, x):
        batch_size = x.size(0)

        x = self.in_conv(x)  # H/2
        x = self.res_1(x)
        x = self.down_1(x)  # H/4
        x = self.res_2(x)
        # x = self.down_2(x)  # H/8
        # x = self.res_3(x)
        # x = self.down_3(x)  # H/16
        x = self.out_conv(x)

        return x


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    x = torch.randn([10, 1, 64, 64, 64])

    net = Encoder_Down_4(channel_in=1, channel_out=128)
    print(net(x).shape)
