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
from brt.jit import make_jit_kernel

import models.archs.arch_util as arch_util
from models.archs.classSR_rcan_arch import classSR_3class_rcan
from models.archs.RCAN_arch import RCAN, ResidualGroup, RCAB, CALayer, Upsampler


class fused_classSR_3class_rcan_net(nn.Module):
    def __init__(self, raw: classSR_3class_rcan, subnet_bs: Tuple[int] = (34, 38, 29)):
        super(fused_classSR_3class_rcan_net, self).__init__()
        self.upscale = 4
        self.subnet_bs = subnet_bs
        self.classifier = raw.classifier
        self.scatter_router = ScatterRouter(
            protocol_type="topk", protocol_kwargs={"top_k": 1}
        )
        subnets = [raw.net1, raw.net2, raw.net3]
        self.fused_rcan = FusedRCAN(subnets, subnet_bs)
        # self.fused_head = FusedLayer(
        #     [subnet.head_conv for subnet in subnets],
        #     [[bs, 3, 32, 32] for bs in subnet_bs],
        #     [
        #         [subnet_bs[0], 16, 32, 32],
        #         [subnet_bs[1], 36, 32, 32],
        #         [subnet_bs[2], 56, 32, 32],
        #     ],
        # )
        self.gather_router = GatherRouter(fabric_type="combine")

    def forward(self, x, is_train=False):

        weights = self.classifier(x)
        sr_xs = self.scatter_router(x, weights)
        real_bs = [srx.shape[0] for srx in sr_xs]
        proto_info = [collect_proto_attr_stack(srx) for srx in sr_xs]
        sr_xs_padding = [
            srx.resize_([bs, *srx.shape[1:]]) for bs, srx in zip(self.subnet_bs, sr_xs)
        ]
        # xs = self.fused_head(sr_xs_padding)
        # xs = self.fused_bodys(xs)
        # # xs = self.fused_tail(xs)
        # xs = [self.tail_convs[i](xs[i]) for i in range(3)]
        # for i in range(3):
        #     xs[i] = init_proto_tensor(xs[i][:real_bs[i]], *proto_info[i])
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
    # Support Conv2dBias, Conv2dBiasReLU, Conv2dBiasSigmoid and AdaptiveAvgPool2d
    def __init__(
        self,
        models: Union[List[nn.Module], List[nn.Sequential]],
        input_shapes: List[torch.Size],
        output_shapes: List[torch.Size],
    ):
        super().__init__()
        models = nn.ModuleList(models)
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.num_submodels = len(models)
        assert len(input_shapes) == self.num_submodels
        assert len(output_shapes) == self.num_submodels

        for i, model in enumerate(models):
            for name, tensor in model.named_parameters(f"m{i}"):
                self.register_parameter(name.replace(".", "_"), tensor)
            for name, tensor in model.named_buffers(f"m{i}"):
                self.register_buffer(name.replace(".", "_"), tensor)
        sample_inputs = [torch.randn(shp).cuda() for shp in input_shapes]
        self.fused_kernel = make_jit_kernel(
            models, sample_inputs, opt_level="horiz_fuse"
        )

        if isinstance(models[0], nn.Conv2d) and models[0].bias is not None:
            self.module_name = "Conv2dBias"
        elif isinstance(models[0], nn.Sequential):
            if isinstance(models[0][0], nn.Conv2d) and models[0][0].bias is not None:
                if isinstance(models[0][1], nn.ReLU):
                    self.module_name = "Conv2dBiasReLU"
                elif isinstance(models[0][1], nn.Sigmoid):
                    self.module_name = "Conv2dBiasSigmoid"
                else:
                    raise NotImplementedError(f"{models}")
        elif isinstance(models[0], nn.AdaptiveAvgPool2d):
            self.module_name = "AdaptiveAvgPool2d"
        else:
            self.module_name = "ERROR"
            raise NotImplementedError(f"{models}")

        self.inputs_templete = {}
        self.inputs_templete["forward"] = []
        if "Conv2d" in self.module_name:
            for i in range(self.num_submodels):
                self.inputs_templete["forward"].extend(
                    [
                        None,
                        self.get_parameter(f"m{i}_weight"),
                        None,
                        self.get_parameter(f"m{i}_bias"),
                    ]
                )
            self.input_indices = [i * 4 for i in range(self.num_submodels)]
            self.output_indices = [i * 4 + 2 for i in range(self.num_submodels)]
        elif self.module_name == "AdaptiveAvgPool2d":
            for i in range(self.num_submodels):
                self.inputs_templete["forward"].extend(
                    [None, None,]
                )
            self.input_indices = [i * 2 for i in range(self.num_submodels)]
            self.output_indices = [i * 2 + 1 for i in range(self.num_submodels)]
        else:
            raise NotImplementedError(f"{self.module_name}")
        self.forward(sample_inputs)

    def forward(self, inputs: List[torch.Tensor]):
        for i in range(self.num_submodels):
            self.inputs_templete["forward"][self.input_indices[i]] = inputs[i]
            self.inputs_templete["forward"][self.output_indices[i]] = torch.empty(
                self.output_shapes[i], device="cuda"
            )
        self.fused_kernel(*self.inputs_templete["forward"])
        outputs = [
            self.inputs_templete["forward"][index] for index in self.output_indices
        ]
        for i in range(self.num_submodels):
            self.inputs_templete["forward"][self.input_indices[i]] = None
            self.inputs_templete["forward"][self.output_indices[i]] = None
        return outputs

    def extra_repr(self):
        return self.module_name


class FusedCALayer(nn.Module):
    def __init__(
        self,
        models: List[CALayer],
        bs: List[int],
        # output_shapes: List[torch.Size],
    ) -> None:
        super().__init__()
        self.avg_pool = FusedLayer(
            # AdaptiveAvgPool2d
            [m.avg_pool for m in models],
            input_shapes=[[n, m.channel, 32, 32] for n, m in zip(bs, models)],
            output_shapes=[[n, m.channel, 1, 1] for n, max in zip(bs, models)],
        )
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

    def foward(self, x: List[torch.Tensor]):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return [xx * yy for xx, yy in zip(x, y)]


class FusedRCAB(nn.Module):
    def __init__(self, models: List[RCAB], bs: List[int],) -> None:
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
        self, models: List[ResidualGroup], bs: List[int], n_resblocks: int = 20,
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

    def forward(self, x: torch.Tensor):
        res = self.body(x)
        return [rr + xx for rr, xx in zip(res, x)]


class FusedUpsampler(nn.Sequential):
    def __init__(self, models: List[Upsampler], bs: List[int],) -> None:
        # assert scale == 4
        super().__init__(
            FusedLayer(
                [m[0] for m in models],
                input_shapes=[[n, m.n_feat, 32, 32] for n, m in zip(bs, models)],
                output_shapes=[[n, m.n_feat * 4, 32, 32] for n, m in zip(bs, models)],
            ),
            FusedLayer(
                [m[1] for m in models],
                input_shapes=[[n, m.n_feat * 4, 32, 32] for n, m in zip(bs, models)],
                output_shapes=[[n, m.n_feat, 64, 64] for n, m in zip(bs, models)],
            ),
            FusedLayer(
                [m[2] for m in models],
                input_shapes=[[n, m.n_feat, 64, 64] for n, m in zip(bs, models)],
                output_shapes=[[n, m.n_feat * 4, 64, 64] for n, m in zip(bs, models)],
            ),
            FusedLayer(
                [m[3] for m in models],
                input_shapes=[[n, m.n_feat * 4, 32, 32] for n, m in zip(bs, models)],
                output_shapes=[[n, m.n_feat, 128, 128] for n, m in zip(bs, models)],
            ),
        )


class FusedRCAN(nn.Module):
    def __init__(
        self, models: List[RCAN], bs: List[int], n_resgroups: int = 10
    ) -> None:
        super().__init__()
        self.sub_mean = FusedLayer(
            [m.sub_mean for m in models],
            input_shapes=[[n, 3, 32, 32] for n in bs],
            output_shapes=[[n, 3, 32, 32] for n in bs],
        )
        self.head = FusedLayer(
            [m.head for m in models],
            input_shapes=[[n, 3, 32, 32] for n in bs],
            output_shapes=[[n, m.n_feats, 32, 32] for n, m in zip(bs, models)],
        )
        self.body = nn.Sequential(
            *[
                FusedResidualGroup([m.body[i] for m in models], bs)
                for i in range(n_resgroups)
            ],
            FusedLayer(
                [m.body[10] for m in models],
                input_shapes=[[n, m.n_feat, 32, 32] for n, m in zip(bs, models)],
                output_shapes=[[n, m.n_feat, 32, 32] for n, m in zip(bs, models)],
            )
        )
        self.tail = nn.Sequential(
            FusedUpsampler([m.tail[0] for m in models], bs),
            FusedLayer(
                [m.body[10] for m in models],
                input_shapes=[[n, m.n_feat, 128, 128] for n, m in zip(bs, models)],
                output_shapes=[[n, 3, 128, 128] for n, m in zip(bs, models)],
            )
        )
        self.add_mean = FusedLayer(
            [m.add_mean for m in models],
            input_shapes=[[n, 3, 32, 32] for n in bs],
            output_shapes=[[n, 3, 32, 32] for n in bs],
        )
    
    def forward(self, x: List[torch.Tensor]):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res = [rr + xx for rr, xx in zip(res, x)]

        x = self.tail(res)
        x = self.add_mean(x)

        return x


