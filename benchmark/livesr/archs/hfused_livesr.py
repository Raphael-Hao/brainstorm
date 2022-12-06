from typing import List
import itertools

import torch
from torch import nn

from brt.jit import make_jit_module
from brt.router import ScatterRouter, GatherRouter
from brt.runtime.proto_tensor import (
    make_proto_tensor_from,
    to_torch_tensor,
    collect_proto_attr_stack,
    init_proto_tensor,
)

from archs.livesr import LiveSR, Classifier
from archs.nas_mdsr import SingleNetwork as NAS, ResBlock, Upsampler

# from archs.fuse import TunedKernel, FusedLayer
from archs.conv2d_mul_add import Conv2dMulAdd


class hFusedLiveSR(nn.Module):
    def __init__(self, raw: LiveSR, subnet_bs: List[int]):
        super().__init__()
        self.subnet_bs = subnet_bs
        self.n_subnets = raw.n_subnets
        self.subnet_num_block = raw.subnet_num_block
        self.num_feature = raw.num_feature
        self.classifier = raw.classifier
        self.scatter = ScatterRouter()
        self.subnets = hFusedNAS(raw.subnets, subnet_bs, self.subnet_num_block)
        self.gather = GatherRouter()

    def forward(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        """@param x: Tensor with shape [N, 3, 32, 32]"""
        scores = self.classifier(inputs)
        # print(scores)
        scattered = self.scatter(inputs, scores)
        real_bs = [sr.shape[0] for sr in scattered]
        proto_info = [collect_proto_attr_stack(srx) for srx in scattered]
        scattered_padding = [
            srx.resize_([bs, *srx.shape[1:]])
            for bs, srx in zip(self.subnet_bs, scattered)
        ]
        subnet_outputs = self.subnets(scattered_padding, self.subnet_num_block)
        for i in range(self.n_subnets):
            subnet_outputs[i] = init_proto_tensor(
                subnet_outputs[i][: real_bs[i]], *proto_info[i]
            )
        gathered = self.gather(subnet_outputs)
        return gathered


class hFusedNAS(nn.Module):
    def __init__(self, raw: List[NAS], subnet_bs: List[int], num_block: int):
        super().__init__()
        self.subnet_bs = subnet_bs
        self.subnet_num_block = num_block
        self.scale = raw[0].scale
        # self.head = FusedLayer(
        #     [m.head for m in raw],
        #     input_shapes=[
        #         [bs, m.num_channel, 32, 32] for m, bs in zip(raw, self.subnet_bs)
        #     ],
        #     output_shapes=[
        #         [bs, m.num_feature, 32, 32] for m, bs in zip(raw, self.subnet_bs)
        #     ],
        # )
        self.head = nn.ModuleList([m.head for m in raw])
        self.body = nn.ModuleList(
            [
                hFusedResBlock(
                    [m.body[i][0] for m in raw],
                    subnet_bs,
                )
                for i in range(self.subnet_num_block)
            ]
        )
        # self.body_end = FusedLayer(
        #     [m.body_end for m in raw],
        #     input_shapes=[
        #         [bs, m.num_feature, 32, 32] for m, bs in zip(raw, self.subnet_bs)
        #     ],
        #     output_shapes=[
        #         [bs, m.num_feature, 32, 32] for m, bs in zip(raw, self.subnet_bs)
        #     ],
        # )
        self.body_end = make_jit_module(
            nn.ModuleList(m.body_end for m in raw),
            sample_inputs=[
                torch.empty(bs, m.num_feature, 32, 32).cuda()
                for m, bs in zip(raw, self.subnet_bs)
            ],
            opt_level="horiz_fuse",
        )
        if self.scale > 1:
            self.upscale = hFusedUpsampler([m.upscale[0] for m in raw], subnet_bs)
        # self.tail = FusedLayer(
        #     [m.tail for m in raw],
        #     input_shapes=[
        #         [bs, m.num_feature, 32, 32] for m, bs in zip(raw, self.subnet_bs)
        #     ],
        #     output_shapes=[
        #         [bs, m.num_feature, 32, 32] for m, bs in zip(raw, self.subnet_bs)
        #     ],
        # )
        self.tail = nn.ModuleList([m.tail for m in raw])

    def forward(self, x: List[torch.Tensor], idx: int) -> List[torch.Tensor]:
        # x = self.head(x)
        x = [m(xx) for m, xx in zip(self.head, x)]
        res = x
        for i in range(idx):
            res = self.body[i](res)
        res = self.body_end(*res)
        res = [rr + xx for rr, xx in zip(res, x)]
        if self.scale > 1:
            x = self.upscale(res)
        else:
            x = res
        # x = self.tail(x)
        x = [m(xx) for m, xx in zip(self.tail, x)]

        return x


class hFusedResBlock(nn.Module):
    def __init__(self, raw: List[ResBlock], subnet_bs: List[int]):
        super().__init__()
        # self.body = nn.Sequential(
        #     *[
        #         FusedLayer(
        #             [nn.Sequential(m.body[0], m.body[1]) for m in raw],
        #             input_shapes=[
        #                 [bs, m.body[0].in_channels, 32, 32]
        #                 for m, bs in zip(raw, subnet_bs)
        #             ],
        #             output_shapes=[
        #                 [bs, m.body[0].out_channels, 32, 32]
        #                 for m, bs in zip(raw, subnet_bs)
        #             ],
        #         ),
        #         FusedLayer(
        #             [m.body[2] for m in raw],
        #             input_shapes=[
        #                 [bs, m.body[2].in_channels, 32, 32]
        #                 for m, bs in zip(raw, subnet_bs)
        #             ],
        #             output_shapes=[
        #                 [bs, m.body[2].out_channels, 32, 32]
        #                 for m, bs in zip(raw, subnet_bs)
        #             ],
        #         ),
        #     ]
        # )
        self.body_0 = make_jit_module(
            nn.ModuleList(nn.Sequential(m.body[0], m.body[1]) for m in raw),
            sample_inputs=[
                torch.empty(bs, m.body[0].in_channels, 32, 32).cuda()
                for m, bs in zip(raw, subnet_bs)
            ],
            opt_level="horiz_fuse",
        )
        self.body_1 = make_jit_module(
            nn.ModuleList(Conv2dMulAdd(m.body[2], m.res_scale) for m in raw),
            sample_inputs=[
                [
                    torch.empty(bs, m.body[2].in_channels, 32, 32).cuda(),
                    torch.empty(bs, m.body[2].in_channels, 32, 32).cuda(),
                ]
                for m, bs in zip(raw, subnet_bs)
            ],
            opt_level="horiz_fuse",
        )
        self.res_scales = [m.res_scale for m in raw]

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        # if self.res_scales != 1:
        #    res = [xx.mul(ss) for xx, ss in zip(self.body(x), self.res_scales)]
        # else:
        #     res = self.body(x)
        y = self.body_0(*x)
        y = self.body_1(*itertools.chain.from_iterable(zip(y, x)))
        return y


class hFusedUpsampler(nn.Module):
    def __init__(self, raw: List[Upsampler], subnet_bs: List[int]):
        super().__init__()
        self.upsampler = nn.ModuleList(
            [
                # FusedLayer(
                #     [m[0] for m in raw],
                #     input_shapes=[[bs, m[0].in_channels, 32, 32] for m, bs in zip(raw, subnet_bs)],
                #     output_shapes=[[bs, m[0].out_channels, 32, 32] for m, bs in zip(raw, subnet_bs)],
                # ),
                nn.ModuleList([m[0] for m in raw]),
                nn.ModuleList([m[1] for m in raw]),
                # FusedLayer(
                #     [m[2] for m in raw],
                #     input_shapes=[[bs, m[2].in_channels, 32, 32] for m, bs in zip(raw, subnet_bs)],
                #     output_shapes=[[bs, m[2].out_channels, 32, 32] for m, bs in zip(raw, subnet_bs)],
                # ),
                nn.ModuleList([m[2] for m in raw]),
                nn.ModuleList([m[3] for m in raw]),
            ]
        )

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        # x = self.upsampler[0](x)
        x = [m(xx) for m, xx in zip(self.upsampler[0], x)]
        x = [m(xx) for m, xx in zip(self.upsampler[1], x)]
        # x = self.upsampler[2](x)
        x = [m(xx) for m, xx in zip(self.upsampler[2], x)]
        x = [m(xx) for m, xx in zip(self.upsampler[3], x)]
        return x
