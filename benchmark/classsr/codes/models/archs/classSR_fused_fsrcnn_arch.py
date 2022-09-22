import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
import torch

from models.archs.Fused_FSRCNN_arch import Fused_FSRCNN_net
from models.archs.classSR_fsrcnn_arch import classSR_3class_fsrcnn_net

from typing import Type, List, Union, Tuple

from brt.runtime.proto_tensor import (
    make_proto_tensor_from,
    to_torch_tensor,
    collect_proto_attr_stack,
    init_proto_tensor,
)

from brt.router import ScatterRouter, GatherRouter


class classSR_3class_fused_fsrcnn_net(nn.Module):
    def __init__(
        self, raw: classSR_3class_fsrcnn_net, subnet_bs: Tuple[int] = (34, 38, 29)
    ):
        super(classSR_3class_fused_fsrcnn_net, self).__init__()
        self.subnet_bs = subnet_bs
        self.upscale = raw.upscale
        self.classifier = raw.classifier
        self.scatter_router = ScatterRouter(
            protocol_type="topk", protocol_kwargs={"top_k": 1}
        )
        self.net1 = Fused_FSRCNN_net(raw.net1, subnet_bs[0])
        self.net2 = Fused_FSRCNN_net(raw.net2, subnet_bs[1])
        self.net3 = Fused_FSRCNN_net(raw.net3, subnet_bs[2])
        self.gather_router = GatherRouter(fabric_type="combine")

    def forward(self, x, is_train=False):

        weights = self.classifier(x)
        sr_xs = self.scatter_router(x, weights)
        real_bs = [srx.shape[0] for srx in sr_xs] 
        proto_info = [collect_proto_attr_stack(srx) for srx in sr_xs]
        sr_xs = [srx.resize_([bs, *srx.shape[1:]]) for bs, srx in zip(self.subnet_bs, sr_xs)]
        y = [self.net1(sr_xs[0]), self.net2(sr_xs[1]), self.net3(sr_xs[2])]
        for i in range(3):
            y[i] = init_proto_tensor(y[i][:real_bs[i]], *proto_info[i])
        gr_xs = self.gather_router(y)
        return gr_xs, [yy.shape[0] for yy in y]


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
