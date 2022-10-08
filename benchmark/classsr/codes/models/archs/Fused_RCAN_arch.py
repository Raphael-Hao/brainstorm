from typing import Type, List, Union, Tuple

import torch
import torch.nn as nn

from models.archs.RCAN_arch import RCAN, ResidualGroup, RCAB, CALayer, Upsampler

from brt.jit import make_jit_kernel


class FusedKernel(nn.Module):
    # Support Conv2dBias, Conv2dBiasReLU, Conv2dBiasSigmoid and AdaptiveAvgPool2d
    def __init__(
        self,
        model: Union[nn.Module, nn.Sequential],
        input_shapes: torch.Size,
    ):
        super().__init__()
        if isinstance(model, nn.Sequential) and len(model) == 1:
            model = model[0]
        self.input_shapes = input_shapes
        assert len(input_shapes) == self.num_submodels

        for name, tensor in model.named_parameters():
            self.register_parameter(name.replace(".", "_"), tensor)
        for name, tensor in model.named_buffers():
            self.register_buffer(name.replace(".", "_"), tensor)
        sample_inputs = [torch.randn(shp).cuda() for shp in input_shapes]
        self.fused_kernel = make_jit_kernel(model, sample_inputs)

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
        elif self.module_name in ["Conv2dBiasReLU", "Conv2dBiasSigmoid"]:
            for i in range(self.num_submodels):
                self.inputs_templete["forward"].extend(
                    [
                        None,
                        self.get_parameter(f"m{i}_0_weight"),
                        None,
                        self.get_parameter(f"m{i}_0_bias"),
                    ]
                )
            self.input_indices = [i * 4 for i in range(self.num_submodels)]
            self.output_indices = [i * 4 + 2 for i in range(self.num_submodels)]
        else:
            raise NotImplementedError(f"{self.module_name}")
        self.forward(sample_inputs)

    def forward(self, inputs: torch.Tensor):
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
        model: CALayer,
        bs: int,
    ) -> None:
        super().__init__()
        self.avg_pool = model.avg_pool
        self.conv_du = nn.Sequential(
            FusedKernel(
                # Conv2dBiasReLU
                nn.Sequential(model.conv_du[0], model.conv_du[1]),
                input_shapes=[bs, model.channel, 1, 1],
            ),
            FusedKernel(
                # Conv2dBiasSigmoid
                nn.Sequential(model.conv_du[2], model.conv_du[3]),
                input_shapes=[
                    [n, m.channel // m.reduction, 1, 1] for n, m in zip(bs, model)
                ],
            ),
        )

    def foward(self, x: torch.Tensor):
        y = [subpool(xx) for subpool, xx in zip(self.avg_pool, x)]
        y = self.conv_du(y)
        return [xx * yy for xx, yy in zip(x, y)]


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
                input_shapes=[bs, model.n_feat, 32, 32],
            ),
            FusedKernel(
                model.body[2],
                input_shapes=[bs, model.n_feat, 32, 32],
            ),
            FusedCALayer(model.body[3], bs),
        )

    def forward(self, x: torch.Tensor):
        res = self.body(x)
        return [rr + xx for rr, xx in zip(res, x)]


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
                input_shapes=[bs, model.n_feat, 32, 32],
            ),
        )

    def forward(self, x: torch.Tensor):
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
                input_shapes=[bs, model.n_feat, 32, 32],
            ),
            FusedKernel(
                model[1],
                input_shapes=[bs, model.n_feat * 4, 32, 32],
            ),
            FusedKernel(
                model[2],
                input_shapes=[bs, model.n_feat, 64, 64],
            ),
            FusedKernel(
                model[3],
                input_shapes=[bs, model.n_feat * 4, 32, 32],
            ),
        )


class FusedRCAN(nn.Module):
    def __init__(self, model: RCAN, bs: int, n_resgroups: int = 10) -> None:
        super().__init__()
        self.sub_mean = FusedKernel(
            model.sub_mean,
            input_shapes=[[n, 3, 32, 32] for n in bs],
        )
        self.head = FusedKernel(
            model.head,
            input_shapes=[[n, 3, 32, 32] for n in bs],
        )
        self.body = nn.Sequential(
            *[FusedResidualGroup(model.body[i], bs) for i in range(n_resgroups)],
            FusedKernel(
                model.body[10],
                input_shapes=[bs, model.n_feat, 32, 32],
            ),
        )
        self.tail = nn.Sequential(
            FusedUpsampler(model.tail[0], bs),
            FusedKernel(
                model.body[10],
                input_shapes=[bs, model.n_feat, 128, 128],
            ),
        )
        self.add_mean = FusedKernel(
            model.add_mean,
            input_shapes=[[n, 3, 32, 32] for n in bs],
        )

    def forward(self, x: torch.Tensor):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res = [rr + xx for rr, xx in zip(res, x)]

        x = self.tail(res)
        x = self.add_mean(x)

        return x
