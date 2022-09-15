import functools
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
import torch

# from models.archs.FSRCNN_arch import FSRCNN_net
import numpy as np
import time

from typing import Type, List

from brt.router import ScatterRouter, GatherRouter
from brt.jit import make_jit_kernel 

from .classSR_fsrcnn_arch import classSR_3class_fsrcnn_net


class classSR_3class_fused_fsrcnn_net(nn.Module):
    def __init__(self, raw: classSR_3class_fsrcnn_net, subnet_bs: List[int]):
        super(classSR_3class_fused_fsrcnn_net, self).__init__()
        self.upscale = 4
        self.classifier = raw.classifier
        self.scatter_router = ScatterRouter(
            protocol_type="topk", protocol_kwargs={"top_k": 1}
        )
        subnets = [raw.net1, raw.net2, raw.net3]
        self.fused_head = FusedLayer(
            [subnet.head_conv for subnet in subnets],
            [[bs, 3, 32, 32] for bs in subnet_bs],
        )
        self.fused_bodys = [
            FusedLayer(
                [subnet.body_conv[0] for subnet in subnets],
                [[bs, 16, 32, 32] for bs in subnet_bs],
            ),
            FusedLayer(
                [subnet.body_conv[1] for subnet in subnets],
                [[bs, 12, 32, 32] for bs in subnet_bs],
            ),
            FusedLayer(
                [subnet.body_conv[2] for subnet in subnets],
                [[bs, 12, 32, 32] for bs in subnet_bs],
            ),
            FusedLayer(
                [subnet.body_conv[3] for subnet in subnets],
                [[bs, 12, 32, 32] for bs in subnet_bs],
            ),
            FusedLayer(
                [
                    nn.Sequential(subnet.body_conv[4], subnet.body_conv[5])
                    for subnet in subnets
                ],
                [[bs, 12, 32, 32] for bs in subnet_bs],
            ),
            FusedLayer(
                [subnet.body_conv[6] for subnet in subnets],
                [[bs, 12, 32, 32] for bs in subnet_bs],
            ),
        ]
        self.fused_tail = FusedLayer(
            [subnet.tail_conv for subnet in subnets], [[bs, 16, 32, 32] for bs in subnet_bs],
        )
        self.gather_router = GatherRouter(fabric_type="combine")

    def forward(self, x, is_train=False):

        weights = self.classifier(x)
        sr_x = self.scatter_router(x, weights)
        # xs = self.fused_head(sr_x)
        # xs = self.fused_bodys(xs)
        # xs = self.fused_tail(xs)
        # gr_x = self.gather_router(xs)
        # return gr_x, [yy.shape[0] for yy in xs]


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.lastOut = nn.Linear(32, 3)

        # Condtion network
        self.CondNet = nn.Sequential(
            nn.Conv2d(3, 128, 4, 4),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 32, 1),
        )
        self.avgPool2d = nn.AvgPool2d(8)
        arch_util.initialize_weights([self.CondNet], 0.1)

    def forward(self, x):
        # assert x.shape[1:] == torch.Size([3, 32, 32]), x.shape
        out = self.CondNet(x)  # [bs, 32, 8, 8]
        out = self.avgPool2d(out)  # [bs, 32, 1, 1]
        out = out.view(-1, 32)  # [bs, 32]
        out = self.lastOut(out)  # [bs, 3]
        return out


class FusedLayer(nn.Module):
    def __init__(self, models: List[nn.Module], input_shapes: List[torch.Size]):
        super().__init__()
        for model in models:
            for name, tensor in model.named_parameters():
                self.register_parameter(name.replace(".", "_"), tensor)
            for name, tensor in model.named_buffers():
                self.register_buffer(name.replace(".", "_"), tensor)
        sample_inputs = [torch.randn(shp).cuda() for shp in input_shapes]
        self.fused_kernel = make_jit_kernel(
            models, sample_inputs, opt_level="hetero_fuse"
        )
        self.ACTIVE_BLOCKS = [1, 1, 1]
        # self.inputs = [
        #     [[], self]
        # ]
        # elif type == 'body'
        # sample_inputs = [torch.randn((batch_sizes[i], model[i]., 32, 32)) for i in range(models)]

    def forward():
        pass
        # output = ...
        # hetero_fused_kernel(*hetero_fused_inputs, active_blocks=active_blocks)
