import torch.nn as nn
from typing import Type, List, Union, Tuple

import models.archs.arch_util as arch_util
from models.archs.Fused_RCAN_arch import FusedRCAN
from models.archs.classSR_rcan_arch import classSR_3class_rcan_net

from brt.router import ScatterRouter, GatherRouter
from brt.runtime.proto_tensor import (
    make_proto_tensor_from,
    to_torch_tensor,
    collect_proto_attr_stack,
    init_proto_tensor,
)


class classSR_3class_fused_rcan_net(nn.Module):
    def __init__(
        self,
        raw: classSR_3class_rcan_net,
        subnet_bs: Tuple[int] = (27, 50, 28),
        n_resgroups: int = 10,
        n_resblocks: int = 20,
    ):
        super(classSR_3class_fused_rcan_net, self).__init__()
        self.upscale = 4
        self.subnet_bs = subnet_bs
        self.classifier = raw.classifier
        self.scatter_router = ScatterRouter(
            protocol_type="topk", protocol_kwargs={"top_k": 1}
        )
        self.net1 = FusedRCAN(raw.net1, subnet_bs[0], n_resgroups, n_resblocks)
        self.net2 = FusedRCAN(raw.net2, subnet_bs[1], n_resgroups, n_resblocks)
        self.net3 = FusedRCAN(raw.net3, subnet_bs[2], n_resgroups, n_resblocks)
        self.gather_router = GatherRouter(fabric_type="combine")

    def forward(self, x, is_train=False):
        import pdb; pdb.set_trace()

        weights = self.classifier(x.div(255.0))
        sr_xs = self.scatter_router(x, weights)
        real_bs = [srx.shape[0] for srx in sr_xs]
        proto_info = [collect_proto_attr_stack(srx) for srx in sr_xs]
        sr_xs_padding = [
            srx.resize_([bs, *srx.shape[1:]]) for bs, srx in zip(self.subnet_bs, sr_xs)
        ]
        xs = [
            self.net1(sr_xs_padding[0]),
            self.net2(sr_xs_padding[1]),
            self.net3(sr_xs_padding[2]),
        ]
        for i in range(3):
            xs[i] = init_proto_tensor(xs[i][: real_bs[i]], *proto_info[i])
        gr_x = self.gather_router(xs)
        return gr_x, [yy.shape[0] for yy in xs]
