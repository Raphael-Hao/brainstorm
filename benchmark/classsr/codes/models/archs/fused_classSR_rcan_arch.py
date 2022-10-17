import logging
import functools
from typing import Type, List, Union, Tuple
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from brt.router import ScatterRouter, GatherRouter
from brt.runtime.proto_tensor import (
    make_proto_tensor_from,
    to_torch_tensor,
    collect_proto_attr_stack,
    init_proto_tensor,
)

import models.archs.arch_util as arch_util
from models.archs.classSR_rcan_arch import classSR_3class_rcan_net
from models.archs.RCAN_arch import RCAN, ResidualGroup, RCAB, CALayer, Upsampler
from models.archs.fuse import FusedKernel, FusedLayer, set_objective_func

logger = logging.getLogger("ClassSR")
logger.setLevel(logging.INFO)


class fused_classSR_3class_rcan_net(nn.Module):
    def __init__(
        self,
        raw: classSR_3class_rcan_net,
        subnet_bs: Tuple[int] = (34, 38, 29),
        objective_func="fastest",
        n_resgroups: int = 10,
        n_resblocks: int = 20,
    ):
        super(fused_classSR_3class_rcan_net, self).__init__()
        with set_objective_func(objective_func):
            self.upscale = 4
            self.subnet_bs = subnet_bs
            self.classifier = raw.classifier
            self.scatter_router = ScatterRouter(
                protocol_type="topk", protocol_kwargs={"top_k": 1}
            )
            subnets = [raw.net1, raw.net2, raw.net3]
            self.fused_rcan = FusedRCAN(subnets, subnet_bs, n_resgroups, n_resblocks)
            self.gather_router = GatherRouter(fabric_type="combine")

    def forward(self, x: torch.Tensor, is_train: bool = False):

        weights = self.classifier(x.div(255.0))
        sr_xs = self.scatter_router(x, weights)
        real_bs = [srx.shape[0] for srx in sr_xs]
        proto_info = [collect_proto_attr_stack(srx) for srx in sr_xs]
        sr_xs_padding = [
            srx.resize_([bs, *srx.shape[1:]]) for bs, srx in zip(self.subnet_bs, sr_xs)
        ]
        xs = self.fused_rcan(sr_xs_padding)
        for i in range(3):
            xs[i] = init_proto_tensor(xs[i][: real_bs[i]], *proto_info[i])
        gr_x = self.gather_router(xs)
        return gr_x, [yy.shape[0] for yy in xs]


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

    def forward(self, x: torch.Tensor):
        # assert x.shape[1:] == torch.Size([3, 32, 32]), x.shape
        out = self.CondNet(x)  # [bs, 32, 8, 8]
        out = self.avgPool2d(out)  # [bs, 32, 1, 1]
        out = out.view(-1, 32)  # [bs, 32]
        out = self.lastOut(out)  # [bs, 3]
        return out


class FusedCALayer(nn.Module):
    def __init__(
        self,
        models: List[CALayer],
        bs: List[int],
        # output_shapes: List[torch.Size],
    ) -> None:
        super().__init__()
        self.avg_pool = [m.avg_pool for m in models]
        self.conv_du = nn.Sequential(
            FusedLayer(
                # Conv2dBiasReLU
                [nn.Sequential(m.conv_du[0], m.conv_du[1]) for m in models],
                input_shapes=[[n, m.channel, 1, 1] for n, m in zip(bs, models)],
                output_shapes=[
                    [n, m.channel // m.reduction, 1, 1] for n, m in zip(bs, models)
                ],
            ),
            FusedLayer(
                # Conv2dBiasSigmoid
                [nn.Sequential(m.conv_du[2], m.conv_du[3]) for m in models],
                input_shapes=[
                    [n, m.channel // m.reduction, 1, 1] for n, m in zip(bs, models)
                ],
                output_shapes=[[n, m.channel, 1, 1] for n, m in zip(bs, models)],
            ),
        )

    def forward(self, x: List[torch.Tensor]):
        y = [subpool(xx) for subpool, xx in zip(self.avg_pool, x)]
        y = self.conv_du(y)
        return [xx * yy for xx, yy in zip(x, y)]


class FusedRCAB(nn.Module):
    def __init__(
        self,
        models: List[RCAB],
        bs: List[int],
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            FusedLayer(
                [nn.Sequential(m.body[0], m.body[1]) for m in models],
                input_shapes=[[n, m.n_feat, 32, 32] for n, m in zip(bs, models)],
                output_shapes=[[n, m.n_feat, 32, 32] for n, m in zip(bs, models)],
            ),
            FusedLayer(
                [m.body[2] for m in models],
                input_shapes=[[n, m.n_feat, 32, 32] for n, m in zip(bs, models)],
                output_shapes=[[n, m.n_feat, 32, 32] for n, m in zip(bs, models)],
            ),
            FusedCALayer([m.body[3] for m in models], bs),
        )

    def forward(self, x: List[torch.Tensor]):
        res = self.body(x)
        return [rr + xx for rr, xx in zip(res, x)]


class FusedResidualGroup(nn.Module):
    def __init__(
        self,
        models: List[ResidualGroup],
        bs: List[int],
        n_resblocks: int = 20,
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            *[FusedRCAB([m.body[i] for m in models], bs) for i in range(n_resblocks)],
            FusedLayer(
                [m.body[n_resblocks] for m in models],
                input_shapes=[[n, m.n_feat, 32, 32] for n, m in zip(bs, models)],
                output_shapes=[[n, m.n_feat, 32, 32] for n, m in zip(bs, models)],
            ),
        )
        logger.info("FusedResidualGroup builded")

    def forward(self, x: List[torch.Tensor]):
        res = self.body(x)
        return [rr + xx for rr, xx in zip(res, x)]


class FusedUpsampler(nn.Sequential):
    def __init__(
        self,
        model: Upsampler,
        bs: int,
    ) -> None:
        # assert scale == 4
        super().__init__(
            FusedKernel(
                model[0],
                input_shape=[bs, model.n_feat, 32, 32],
                output_shape=[bs, model.n_feat * 4, 32, 32],
            ),
            model[1],
            FusedKernel(
                model[2],
                input_shape=[bs, model.n_feat, 64, 64],
                output_shape=[bs, model.n_feat * 4, 64, 64],
            ),
            model[3],
        )


class FusedRCAN(nn.Module):
    def __init__(
        self,
        models: List[RCAN],
        bs: List[int],
        n_resgroups: int = 10,
        n_resblocks: int = 20,
    ) -> None:
        super().__init__()
        self.sub_mean = FusedLayer(
            [m.sub_mean for m in models],
            input_shapes=[[n, 3, 32, 32] for n in bs],
            output_shapes=[[n, 3, 32, 32] for n in bs],
        )
        logger.info("FusedRCAN.sub_mean builded")
        self.head = FusedLayer(
            [m.head for m in models],
            input_shapes=[[n, 3, 32, 32] for n in bs],
            output_shapes=[[n, m.n_feat, 32, 32] for n, m in zip(bs, models)],
        )
        logger.info("FusedRCAN.head builded")
        self.body = nn.Sequential(
            *[
                FusedResidualGroup([m.body[i] for m in models], bs, n_resblocks)
                # for i in range(1)
                for i in range(n_resgroups)
            ],
            FusedLayer(
                [m.body[n_resgroups] for m in models],
                input_shapes=[[n, m.n_feat, 32, 32] for n, m in zip(bs, models)],
                output_shapes=[[n, m.n_feat, 32, 32] for n, m in zip(bs, models)],
            ),
        )
        logger.info("FusedRCAN.body builded")
        self.tail = nn.Sequential(
            FusedUpsampler([m.tail[0] for m in models], bs),
            FusedLayer(
                [m.tail[1] for m in models],
                input_shapes=[[n, m.n_feat, 128, 128] for n, m in zip(bs, models)],
                output_shapes=[[n, 3, 128, 128] for n in bs],
            ),
        )
        logger.info("FusedRCAN.tail builded")
        self.add_mean = FusedLayer(
            [m.add_mean for m in models],
            input_shapes=[[n, 3, 128, 128] for n in bs],
            output_shapes=[[n, 3, 128, 128] for n in bs],
        )
        logger.info("FusedRCAN.add_mean builded")

    def forward(self, x: List[torch.Tensor]):

        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res = [rr + xx for rr, xx in zip(res, x)]

        x = self.tail(res)
        x = self.add_mean(x)

        return x
