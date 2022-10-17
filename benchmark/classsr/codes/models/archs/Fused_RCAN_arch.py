import logging
from typing import Type, List, Union, Tuple

import torch
import torch.nn as nn

from models.archs.RCAN_arch import RCAN, ResidualGroup, RCAB, CALayer, Upsampler

from brt.jit import make_jit_kernel

logger = logging.getLogger("ClassSR")
logger.setLevel(logging.DEBUG)


class FusedKernel(nn.Module):
    # Support Conv2dBias, Conv2dBiasReLU, Conv2dBiasSigmoid and AdaptiveAvgPool2d
    def __init__(
        self,
        model: Union[nn.Module, nn.Sequential],
        input_shape: torch.Size,
        output_shape: torch.Size,
    ):
        super().__init__()
        if isinstance(model, nn.Sequential) and len(model) == 1:
            model = model[0]
        self.input_shape = input_shape
        self.output_shape = output_shape

        for name, tensor in model.named_parameters():
            self.register_parameter(name.replace(".", "_"), tensor)
        for name, tensor in model.named_buffers():
            self.register_buffer(name.replace(".", "_"), tensor)
        sample_input = torch.randn(self.input_shape).cuda()
        self.fused_kernel = make_jit_kernel(model, sample_input)

        if isinstance(model, nn.Conv2d) and model.bias is not None:
            self.module_name = "Conv2dBias"
        elif isinstance(model, nn.Sequential):
            if isinstance(model[0], nn.Conv2d) and model[0].bias is not None:
                if len(model) == 1:
                    self.module_name = "Conv2dBias"
                elif isinstance(model[1], nn.ReLU):
                    self.module_name = "Conv2dBiasReLU"
                elif isinstance(model[1], nn.Sigmoid):
                    self.module_name = "Conv2dBiasSigmoid"
                else:
                    raise NotImplementedError(f"{model}")
            else:
                raise NotImplementedError(f"{model}")
        else:
            self.module_name = "ERROR"
            raise NotImplementedError(f"{model}")

        self.inputs_templete = {}
        self.inputs_templete["forward"] = []
        if self.module_name == "Conv2dBias":
            self.inputs_templete["forward"].extend(
                [
                    None,
                    self.get_parameter(f"weight"),
                    None,
                    self.get_parameter(f"bias"),
                ]
            )
        elif self.module_name in ["Conv2dBiasReLU", "Conv2dBiasSigmoid"]:
            self.inputs_templete["forward"].extend(
                [
                    None,
                    self.get_parameter(f"0_weight"),
                    None,
                    self.get_parameter(f"0_bias"),
                ]
            )
        else:
            raise NotImplementedError(f"{self.module_name}")
        self.forward(sample_input)

    def forward(self, input: torch.Tensor):
        self.inputs_templete["forward"][0] = input
        self.inputs_templete["forward"][2] = torch.empty(
            self.output_shape, device="cuda"
        )
        self.fused_kernel(*self.inputs_templete["forward"])
        output = self.inputs_templete["forward"][2]
        self.inputs_templete["forward"][0] = None
        self.inputs_templete["forward"][2] = None
        return output

    def extra_repr(self):
        return self.module_name


class FusedCALayer(nn.Module):
    def __init__(
        self,
        model: CALayer,
        bs: int,
    ) -> None:
        super().__init__()
        self.avg_pool = model.avg_pool
        self.conv_du = nn.Sequential(
            FusedKernel(
                # Conv2dBiasReLU
                nn.Sequential(model.conv_du[0], model.conv_du[1]),
                input_shape=[bs, model.channel, 1, 1],
                output_shape=[bs, model.channel // model.reduction, 1, 1],
            ),
            FusedKernel(
                # Conv2dBiasSigmoid
                nn.Sequential(model.conv_du[2], model.conv_du[3]),
                input_shape=[bs, model.channel // model.reduction, 1, 1],
                output_shape=[bs, model.channel, 1, 1],
            ),
        )

    def forward(self, x: torch.Tensor):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class FusedRCAB(nn.Module):
    def __init__(
        self,
        model: RCAB,
        bs: int,
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            FusedKernel(
                nn.Sequential(model.body[0], model.body[1]),
                input_shape=[bs, model.n_feat, 32, 32],
                output_shape=[bs, model.n_feat, 32, 32],
            ),
            FusedKernel(
                model.body[2],
                input_shape=[bs, model.n_feat, 32, 32],
                output_shape=[bs, model.n_feat, 32, 32],
            ),
            FusedCALayer(model.body[3], bs),
        )

    def forward(self, x: torch.Tensor):
        res = self.body(x)
        return res + x


class FusedResidualGroup(nn.Module):
    def __init__(
        self,
        model: ResidualGroup,
        bs: int,
        n_resblocks: int = 20,
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            *[FusedRCAB(model.body[i], bs) for i in range(n_resblocks)],
            FusedKernel(
                model.body[n_resblocks],
                input_shape=[bs, model.n_feat, 32, 32],
                output_shape=[bs, model.n_feat, 32, 32],
            ),
        )

    def forward(self, x: torch.Tensor):
        res = self.body(x)
        return res + x


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
        self, model: RCAN, bs: int, n_resgroups: int = 10, n_resblocks: int = 20
    ) -> None:
        super().__init__()
        self.sub_mean = FusedKernel(
            model.sub_mean,
            input_shape=[bs, 3, 32, 32],
            output_shape=[bs, 3, 32, 32],
        )
        logger.info("FusedRCAN.sub_mean builded")
        self.head = FusedKernel(
            model.head,
            input_shape=[bs, 3, 32, 32],
            output_shape=[bs, model.n_feat, 32, 32],
        )
        logger.info("FusedRCAN.head builded")
        self.body = nn.Sequential(
            *[
                FusedResidualGroup(model.body[i], bs, n_resblocks)
                for i in range(n_resgroups)
            ],
            # *[FusedResidualGroup(model.body[i], bs) for i in range(1)],
            FusedKernel(
                model.body[n_resgroups],
                input_shape=[bs, model.n_feat, 32, 32],
                output_shape=[bs, model.n_feat, 32, 32],
            ),
        )
        logger.info("FusedRCAN.body builded")
        self.tail = nn.Sequential(
            FusedUpsampler(model.tail[0], bs),
            FusedKernel(
                model.tail[1],
                input_shape=[bs, model.n_feat, 128, 128],
                output_shape=[bs, 3, 128, 128],
            ),
        )
        logger.info("FusedRCAN.tail builded")
        self.add_mean = FusedKernel(
            model.add_mean,
            input_shape=[bs, 3, 128, 128],
            output_shape=[bs, 3, 128, 128],
        )
        logger.info("FusedRCAN.add_mean builded")

    def forward(self, x: torch.Tensor):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res = res + x

        x = self.tail(res)
        x = self.add_mean(x)

        return x
