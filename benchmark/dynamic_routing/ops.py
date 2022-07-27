# Routinng candidates for dynamic networks with calculated FLOPs,
# modified Search Space in DARTS to have different input and output channels.
# @author: yanwei.li
from collections import namedtuple

import torch.nn as nn
import torch

OPS = {
    "skip_connect": lambda C_in, C_out, stride, norm_layer, affine, input_size: Identity(
        C_in, C_out, norm_layer=norm_layer, affine=affine, input_size=input_size
    )
    if stride == 1
    else FactorizedReduce(
        C_in, C_out, norm_layer=norm_layer, affine=affine, input_size=input_size
    ),
    "sep_conv_3x3": lambda C_in, C_out, stride, norm_layer, affine, input_size: SepConv(
        C_in,
        C_out,
        3,
        stride,
        1,
        norm_layer=norm_layer,
        affine=affine,
        input_size=input_size,
    ),
    "sep_conv_5x5": lambda C_in, C_out, stride, norm_layer, affine, input_size: SepConv(
        C_in,
        C_out,
        5,
        stride,
        2,
        norm_layer=norm_layer,
        affine=affine,
        input_size=input_size,
    ),
    "sep_conv_7x7": lambda C_in, C_out, stride, norm_layer, affine, input_size: SepConv(
        C_in,
        C_out,
        7,
        stride,
        3,
        norm_layer=norm_layer,
        affine=affine,
        input_size=input_size,
    ),
}


class ShapeSpec(namedtuple("_ShapeSpec", ["channels", "height", "width", "stride"])):
    """
    A simple structure that contains basic shape specification about a tensor.
    It is often used as the auxiliary inputs/outputs of models,
    to obtain the shape inference ability among pytorch modules.

    Attributes:
        channels:
        height:
        width:
        stride:
    """

    def __new__(cls, *, channels=None, height=None, width=None, stride=None):
        return super().__new__(cls, channels, height, width, stride)


def kaiming_init_module(
    module, a=0, mode="fan_out", nonlinearity="relu", bias=0, distribution="normal"
):
    assert distribution in ["uniform", "normal"]

    for _name, m in module.named_modules():
        if isinstance(m, Conv2dNormAct):
            # TODO: not accessible
            # assert not hasattr(m, "weight")
            # assert not hasattr(m, "bias")
            if hasattr(m, "weight") and m.weight is not None:
                if distribution == "uniform":
                    nn.init.kaiming_uniform_(
                        m.weight, a=a, mode=mode, nonlinearity=nonlinearity
                    )
                elif m:
                    nn.init.kaiming_normal_(
                        m.weight, a=a, mode=mode, nonlinearity=nonlinearity
                    )
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Conv2dNormAct(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        norm=None,
        activation=None,
    ) -> None:
        # print(f"Conv2dNormAct: {in_channels} -> {out_channels}")
        # print(f"kernel_size: {kernel_size}")
        # print(f"stride: {stride}")
        # print(f"padding: {padding}")
        # print(f"dilation: {dilation}")
        # print(f"groups: {groups}")
        # print(f"bias: {bias}")
        # print(f"padding_mode: {padding_mode}")
        # print(f"norm: {norm}")
        # print(f"activation: {activation}")
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SepConv(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        norm_layer,
        affine=True,
        input_size=None,
    ):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            # depth wise
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            # point wise
            Conv2dNormAct(
                C_in,
                C_in,
                kernel_size=1,
                padding=0,
                bias=False,
                norm=nn.SyncBatchNorm(C_in),
                activation=nn.ReLU(),
            ),
            # stack 2 separate depthwise-conv.
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            Conv2dNormAct(
                C_in,
                C_out,
                kernel_size=1,
                padding=0,
                bias=False,
                norm=nn.SyncBatchNorm(C_out),
            ),
        )
        # using Kaiming init
        kaiming_init_module(self.op, mode="fan_in")

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    def __init__(self, C_in, C_out, norm_layer, affine=True, input_size=None):
        super(Identity, self).__init__()
        if C_in == C_out:
            self.change = False
            self.flops = 0.0
        else:
            self.change = True
            self.op = Conv2dNormAct(
                C_in,
                C_out,
                kernel_size=1,
                padding=0,
                bias=False,
                # TODO: norm type
                norm=nn.SyncBatchNorm(C_out),
            )

            # using Kaiming init
            kaiming_init_module(self.op, mode="fan_in")

    def forward(self, x):
        if not self.change:
            return x
        else:
            return self.op(x)


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, norm_layer, affine=True, input_size=None):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.conv_1 = Conv2dNormAct(
            C_in, C_out // 2, 1, stride=2, padding=0, bias=False
        )
        self.conv_2 = Conv2dNormAct(
            C_in, C_out // 2, 1, stride=2, padding=0, bias=False
        )
        self.bn = norm_layer(C_out, affine=affine)

        # using Kaiming init
        for layer in [self.conv_1, self.conv_2]:
            kaiming_init_module(layer, mode="fan_in")

    def forward(self, x):
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out
