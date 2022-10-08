import functools
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
import torch

# from models.archs.FSRCNN_arch import FSRCNN_net
import numpy as np
import time

from typing import Type, List, Union, Tuple

from brt.router import ScatterRouter, GatherRouter
from brt.runtime.proto_tensor import (
    make_proto_tensor_from,
    to_torch_tensor,
    collect_proto_attr_stack,
    init_proto_tensor,
)
from brt.jit import make_jit_kernel

from .classSR_fsrcnn_arch import classSR_3class_fsrcnn_net


class fused_classSR_3class_fsrcnn_net(nn.Module):
    def __init__(
        self, raw: classSR_3class_fsrcnn_net, subnet_bs: Tuple[int] = (34, 38, 29)
    ):
        super(fused_classSR_3class_fsrcnn_net, self).__init__()
        self.upscale = 4
        self.subnet_bs = subnet_bs
        self.classifier = raw.classifier
        self.scatter_router = ScatterRouter(
            protocol_type="topk", protocol_kwargs={"top_k": 1}
        )
        subnets = [raw.net1, raw.net2, raw.net3]
        self.fused_head = FusedLayer(
            [subnet.head_conv for subnet in subnets],
            [[bs, 3, 32, 32] for bs in subnet_bs],
            [
                [subnet_bs[0], 16, 32, 32],
                [subnet_bs[1], 36, 32, 32],
                [subnet_bs[2], 56, 32, 32],
            ],
        )
        self.fused_bodys = nn.Sequential(
            FusedLayer(
                [subnet.body_conv[0] for subnet in subnets],
                [
                    [subnet_bs[0], 16, 32, 32],
                    [subnet_bs[1], 36, 32, 32],
                    [subnet_bs[2], 56, 32, 32],
                ],
                [[bs, 12, 32, 32] for bs in subnet_bs],
            ),
            FusedLayer(
                [subnet.body_conv[1] for subnet in subnets],
                [[bs, 12, 32, 32] for bs in subnet_bs],
                [[bs, 12, 32, 32] for bs in subnet_bs],
            ),
            FusedLayer(
                [subnet.body_conv[2] for subnet in subnets],
                [[bs, 12, 32, 32] for bs in subnet_bs],
                [[bs, 12, 32, 32] for bs in subnet_bs],
            ),
            FusedLayer(
                [subnet.body_conv[3] for subnet in subnets],
                [[bs, 12, 32, 32] for bs in subnet_bs],
                [[bs, 12, 32, 32] for bs in subnet_bs],
            ),
            FusedLayer(
                [
                    nn.Sequential(subnet.body_conv[4], subnet.body_conv[5])
                    for subnet in subnets
                ],
                [[bs, 12, 32, 32] for bs in subnet_bs],
                [[bs, 12, 32, 32] for bs in subnet_bs],
            ),
            FusedLayer(
                [subnet.body_conv[6] for subnet in subnets],
                [[bs, 12, 32, 32] for bs in subnet_bs],
                [
                    [subnet_bs[0], 16, 32, 32],
                    [subnet_bs[1], 36, 32, 32],
                    [subnet_bs[2], 56, 32, 32],
                ],
            ),
        )
        self.tail_convs = [subnet.tail_conv for subnet in subnets]
        # self.fused_tail = FusedLayer(
        #     [subnet.tail_conv for subnet in subnets],
        #     [
        #         [subnet_bs[0], 16, 32, 32],
        #         [subnet_bs[1], 36, 32, 32],
        #         [subnet_bs[2], 56, 32, 32],
        #     ],
        #     [[bs, 3, 128, 128] for bs in subnet_bs],
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
        xs = self.fused_head(sr_xs_padding)
        xs = self.fused_bodys(xs)
        # xs = self.fused_tail(xs)
        xs = [self.tail_convs[i](xs[i]) for i in range(3)]
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

    def forward(self, x):
        # assert x.shape[1:] == torch.Size([3, 32, 32]), x.shape
        out = self.CondNet(x)  # [bs, 32, 8, 8]
        out = self.avgPool2d(out)  # [bs, 32, 1, 1]
        out = out.view(-1, 32)  # [bs, 32]
        out = self.lastOut(out)  # [bs, 3]
        return out


class FusedLayer(nn.Module):
    def __init__(
        self,
        models: Union[List[nn.Module], List[nn.Sequential]],
        input_shapes: List[torch.Size],
        output_shapes: List[torch.Size],
    ):
        super().__init__()
        models = nn.ModuleList(models)
        # print(models)
        self.num_submodels = len(models)
        assert len(input_shapes) == self.num_submodels
        assert len(output_shapes) == self.num_submodels
        for i, model in enumerate(models):
            for name, tensor in model.named_parameters(f"m{i}"):
                self.register_parameter(name.replace(".", "_"), tensor)
            for name, tensor in model.named_buffers(f"m{i}"):
                self.register_buffer(name.replace(".", "_"), tensor)
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        sample_inputs = [torch.randn(shp).cuda() for shp in input_shapes]
        self.fused_kernel = make_jit_kernel(
            models, sample_inputs, opt_level="horiz_fuse"
        )
        # self.ACTIVE_BLOCKS = [1] * self.num_submodels
        # Conv2dBiasPReLU or Conv2dBias or ConvTranspose2dBias
        if isinstance(models[0], nn.Sequential):
            conv2d = models[0][0]
            prelu = models[0][1]
            if (
                isinstance(conv2d, nn.Conv2d)
                and conv2d.bias is not None
                and isinstance(prelu, nn.PReLU)
            ):
                self.module_name = "Conv2dBiasPReLU"
            else:
                raise NotImplementedError(f"{models}")
        elif isinstance(models[0], nn.Conv2d) and models[0].bias is not None:
            self.module_name = "Conv2dBias"
        elif isinstance(models[0], nn.ConvTranspose2d) and models[0].bias is not None:
            self.module_name = "ConvTranspose2dBias"
        else:
            self.module_name = "ERROR"
            raise NotImplementedError(f"{models}")

        self.inputs_templete = {}
        self.inputs_templete["forward"] = []
        if self.module_name == "Conv2dBiasPReLU":
            for i in range(len(models)):
                self.inputs_templete["forward"].extend(
                    [
                        None,
                        self.get_parameter(f"m{i}_0_weight"),
                        None,
                        self.get_parameter(f"m{i}_0_bias"),
                        self.get_parameter(f"m{i}_1_weight")
                        .expand(self.output_shapes[1])
                        .contiguous(),
                    ]
                )
            self.input_indices = [i * 5 for i in range(self.num_submodels)]
            self.output_indices = [i * 5 + 2 for i in range(self.num_submodels)]
        elif (
            self.module_name == "Conv2dBias"
            or self.module_name == "ConvTranspose2dBias"
        ):
            for i in range(len(models)):
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
        # elif self.module_name == "ConvTranspose2dBias":
        else:
            raise NotImplementedError(f"{self.module_name}")
        # self.forward(sample_inputs)

    def forward(self, inputs: List[torch.Tensor]):
        for i in range(self.num_submodels):
            self.inputs_templete["forward"][self.input_indices[i]] = inputs[i]
            self.inputs_templete["forward"][self.output_indices[i]] = torch.empty(
                self.output_shapes[i], device="cuda"
            )
        self.fused_kernel(
            *self.inputs_templete["forward"],
            # active_blocks=self.ACTIVE_BLOCKS,
        )
        outputs = [
            self.inputs_templete["forward"][index] for index in self.output_indices
        ]
        for i in range(self.num_submodels):
            self.inputs_templete["forward"][self.input_indices[i]] = None
            self.inputs_templete["forward"][self.output_indices[i]] = None
        return outputs

        # hetero_fused_kernel(*hetero_fused_inputs, active_blocks=active_blocks)

    def extra_repr(self):
        return self.module_name
