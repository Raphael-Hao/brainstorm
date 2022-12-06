from typing import List


import torch
from torch import nn, fx
from torch.utils import dlpack

from brt.router import ScatterRouter, GatherRouter
from brt.runtime.proto_tensor import (
    make_proto_tensor_from,
    to_torch_tensor,
    collect_proto_attr_stack,
    init_proto_tensor,
)

from archs.livesr import LiveSR, TunedClassifier
from archs.vfused_nas_mdsr import vFusedNAS
from archs.nas_mdsr import SingleNetwork as NAS, ResBlock, Upsampler
from archs.fuse import TunedKernel


class vFusedLiveSR(nn.Module):
    def __init__(self, raw: LiveSR, subnet_bs: List[int]):
        super().__init__()
        self.subnet_bs = subnet_bs
        self.n_subnets = raw.n_subnets
        self.subnet_num_block = raw.subnet_num_block
        self.num_feature = raw.num_feature
        self.classifier = raw.classifier
        self.scatter = ScatterRouter()
        self.subnets = nn.ModuleList(
            vFusedNAS(net, bs) for net, bs in zip(raw.subnets, subnet_bs)
        )
        self.gather = GatherRouter()

    def forward(self, inputs: torch.Tensor):
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
        subnet_outputs = [
            m(x, m.num_block) for m, x in zip(self.subnets, scattered_padding)
        ]
        for i in range(self.n_subnets):
            subnet_outputs[i] = init_proto_tensor(
                subnet_outputs[i][: real_bs[i]], *proto_info[i]
            )
        gathered = self.gather(subnet_outputs)
        return gathered

