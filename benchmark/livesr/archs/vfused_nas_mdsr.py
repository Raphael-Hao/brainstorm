from typing import List

import torch
from torch import nn

from archs.nas_mdsr import SingleNetwork as NAS, ResBlock, Upsampler
from archs.fuse import TunedKernel


class vFusedNAS(nn.Module):
    def __init__(self, raw: NAS, bs: int):
        super().__init__()
        self.num_block = raw.num_block
        self.num_feature = raw.num_feature
        self.num_channel = raw.num_channel
        self.scale = raw.scale
        assert self.scale == 4
        # self.head = TunedKernel(
        #     raw.head,
        #     input_shape=[bs, self.num_channel, 32, 32],
        #     output_shape=[bs, self.num_feature, 32, 32],
        # )
        self.head = raw.head
        self.body = nn.ModuleList([vFusedResBlock(m[0], bs) for m in raw.body])
        self.body_end = TunedKernel(
            raw.body_end,
            input_shape=[bs, self.num_feature, 32, 32],
            output_shape=[bs, self.num_feature, 32, 32],
        )
        if self.scale > 1:
            self.upscale = vFusedUpsampler(raw.upscale[0], bs)
        # self.tail = TunedKernel(raw.tail,
        #     input_shape=[bs, self.num_feature, 128, 128],
        #     output_shape=[bs, self.num_feature, 128, 128],
        # )
        self.tail = raw.tail

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        x = self.head(x)
        res = x
        for i in range(idx):
            res = self.body[i](res)
        res = self.body_end(res)
        res += x
        if self.scale > 1:
            x = self.upscale(res)
        else:
            x = res
        x = self.tail(x)

        return x


class vFusedResBlock(nn.Module):
    def __init__(self, raw: ResBlock, bs: int):
        super().__init__()
        self.body = nn.Sequential(
            *[
                TunedKernel(
                    nn.Sequential(raw.body[0], raw.body[1]),
                    input_shape=[bs, raw.body[0].in_channels, 32, 32],
                    output_shape=[bs, raw.body[0].out_channels, 32, 32],
                ),
                TunedKernel(
                    raw.body[2],
                    input_shape=[bs, raw.body[2].in_channels, 32, 32],
                    output_shape=[bs, raw.body[2].out_channels, 32, 32],
                ),
            ]
        )
        self.res_scale = raw.res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.res_scale != 1:
            res = self.body(x).mul(self.res_scale)
        else:
            res = self.body(x)
        res += x
        return res


class vFusedUpsampler(nn.Module):
    def __init__(self, raw: Upsampler, bs: int):
        super().__init__()
        self.upsampler = nn.Sequential(
            *[
                # TunedKernel(
                #     raw[0],
                #     input_shape=[bs, raw[0].in_channels, 32, 32],
                #     output_shape=[bs, raw[0].out_channels, 32, 32],
                # ),
                raw[0],
                raw[1],
                # TunedKernel(
                #     raw[2],
                #     input_shape=[bs, raw[2].in_channels, 64, 64],
                #     output_shape=[bs, raw[2].out_channels, 64, 64],
                # ),
                raw[2],
                raw[3],
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsampler(x)
