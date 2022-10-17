from typing import Type, List, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from brt.jit import make_jit_kernel


class _ObjectiveFuncContext:
    curr = "fastest"

    def __init__(self, objective_func) -> None:
        self.prev = type(self).curr
        type(self).curr = objective_func

    def __enter__(self) -> None:
        pass

    def __exit__(self) -> None:
        type(self).curr = self.prev


def set_objective_func(objective_func) -> _ObjectiveFuncContext:
    return _ObjectiveFuncContext(objective_func)


def get_objective_func() -> str:
    return _ObjectiveFuncContext.curr


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


class FusedLayer(nn.Module):
    # Support Conv2dBias, Conv2dBiasReLU, Conv2dBiasSigmoid and AdaptiveAvgPool2d
    def __init__(
        self,
        models: Union[List[nn.Module], List[nn.Sequential]],
        input_shapes: List[torch.Size],
        output_shapes: List[torch.Size],
    ):
        super().__init__()
        if not isinstance(models, nn.ModuleList):
            models = nn.ModuleList(models)
        if isinstance(models[0], nn.Sequential) and len(models[0]) == 1:
            models = [m[0] for m in models]
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
            models,
            sample_inputs,
            opt_level="horiz_fuse",
            objective_func=get_objective_func,
        )

        if isinstance(models[0], nn.Conv2d) and models[0].bias is not None:
            self.module_name = "Conv2dBias"
        elif isinstance(models[0], nn.Sequential):
            if isinstance(models[0][0], nn.Conv2d) and models[0][0].bias is not None:
                if len(models[0]) == 1:
                    self.module_name = "Conv2dBias"
                elif isinstance(models[0][1], nn.ReLU):
                    self.module_name = "Conv2dBiasReLU"
                elif isinstance(models[0][1], nn.Sigmoid):
                    self.module_name = "Conv2dBiasSigmoid"
                else:
                    raise NotImplementedError(f"{models}")
            else:
                raise NotImplementedError(f"{models}")
        elif isinstance(models[0], nn.AdaptiveAvgPool2d):
            self.module_name = "AdaptiveAvgPool2d"
        else:
            self.module_name = "ERROR"
            raise NotImplementedError(f"{models}")

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
        elif self.module_name == "AdaptiveAvgPool2d":
            for i in range(self.num_submodels):
                self.inputs_templete["forward"].extend(
                    [
                        None,
                        None,
                    ]
                )
            self.input_indices = [i * 2 for i in range(self.num_submodels)]
            self.output_indices = [i * 2 + 1 for i in range(self.num_submodels)]
        else:
            raise NotImplementedError(f"{self.module_name}")
        self.forward(sample_inputs)

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
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
