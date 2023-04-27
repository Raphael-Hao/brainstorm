# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import builtins
from typing import List, Tuple, Union, Literal, Optional, Type, Dict
import operator

import torch
from torch import nn, fx, autograd

from brt.runtime import log
from brt.jit.modules.atom import AtomModule, JitAtomModule, AtomModuleInputType
from brt.jit.codegen.module import ModuleKernel, ModuleDTypeSizeInByte

logger = log.get_logger(__file__)


class Conv2dElementWiseModule(AtomModule):
    _supported_methods = ["forward"]
    # _input_arg_indices = {"forward": [0]}
    # _output_arg_indices = {"forward": [2]}
    _shared_arg_indices = {"forward": [0, 2]}

    _optional_succeed_module_cls = (
        torch.nn.BatchNorm2d,
        torch.nn.ReLU,
        torch.nn.Sigmoid,
    )
    _optional_succeed_function = (
        operator.add,
        operator.mul,
        torch.add,
        torch.mul,
        torch.Tensor.add,
        torch.Tensor.mul,
    )

    def __init__(self, module: nn.Module):
        super().__init__(module)
        if isinstance(self.module, nn.Conv2d):
            self.module = nn.Sequential(self.module)
        if not isinstance(self.module, fx.GraphModule):
            self.module = fx.symbolic_trace(self.module)
        self._module_inputs = []
        self._conv2d: nn.Conv2d = None
        self._succeed = []
        self._bn: nn.BatchNorm2d = None
        node: fx.Node
        for node in self.module.graph.nodes:
            if node.op == "placeholder":
                self._module_inputs.append(node.name)
            elif node.op == "get_attr":
                pass
            elif node.op == "call_module":
                submodule = self.module.get_submodule(node.target)
                if (
                    isinstance(submodule, nn.Conv2d)
                    and node.args[0].name in self._module_inputs
                ):
                    self._conv2d = submodule
                    self._conv2d_inputs = node.args[0].name
                    conv2d_output = node.name
                elif isinstance(
                    submodule,
                    type(self)._optional_succeed_module_cls,
                ):
                    if isinstance(submodule, nn.BatchNorm2d):
                        self._bn = submodule
                    self._succeed.append(submodule)
                    conv2d_output = node.name
                else:
                    raise RuntimeError
            elif node.op == "call_function":
                if node.target is torch.conv2d:
                    self._conv2d = self
                    self._conv2d_inputs = node.args[0].name
                    conv2d_output = node.name
                elif node.target in type(self)._optional_succeed_function:
                    for farg in node.args:
                        if isinstance(farg, fx.Node) and farg.name == conv2d_output:
                            conv2d_output = node.name
                            self._succeed.append(node)
                else:
                    raise RuntimeError
            elif node.op == "call_method":
                raise RuntimeError
            elif node.op == "output":
                if node.args[0].name != conv2d_output:
                    raise RuntimeError
            else:
                raise RuntimeError

    @classmethod
    def ismodule(cls, module: nn.Module):
        # WARNING: Can't check symbolic traced nn.Conv2d itself
        if isinstance(module, nn.Conv2d):
            return True
        if not isinstance(module, fx.GraphModule):
            try:
                module = fx.symbolic_trace(module)
            except:
                return False
        node: fx.Node
        conv2d_checked = False
        known_vars = set()
        for node in module.graph.nodes:
            if node.op == "placeholder":
                known_vars.add(node.name)
            elif node.op == "get_attr":
                known_vars.add(node.name)
            elif node.op == "call_module":
                if not conv2d_checked:
                    if (
                        isinstance(module.get_submodule(node.target), nn.Conv2d)
                        and node.args[0].name in known_vars
                    ):
                        conv2d_checked = True
                        conv2d_output = node.name
                    else:
                        return False
                else:
                    if (
                        isinstance(
                            module.get_submodule(node.target),
                            cls._optional_succeed_module_cls,
                        )
                        and node.args[0].name == conv2d_output
                    ):
                        conv2d_output = node.name
                    else:
                        return False
            elif node.op == "call_function":
                if not conv2d_checked:
                    if node.target is torch.conv2d and node.args[0].name in known_vars:
                        conv2d_checked = True
                        conv2d_output = node.name
                    else:
                        return False
                else:
                    if node.target in cls._optional_succeed_function:
                        is_following_conv2d = False
                        for farg in node.args:
                            if (
                                isinstance(farg, (int, float))
                                or farg.name in known_vars
                            ):
                                pass
                            elif farg.name == conv2d_output:
                                is_following_conv2d = True
                            else:
                                return False
                        if is_following_conv2d:
                            conv2d_output = node.name
                        else:
                            known_vars.add(node.name)
                    else:
                        return False
            elif node.op == "call_method":
                return False
            elif node.op == "output":
                if node.args[0].name != conv2d_output:
                    return False
            else:
                return False
        return True

    def _make_global_kernel(
        self,
        sample_inputs: AtomModuleInputType,
        method: str,
        objective_func: str = "fastest",
        rank: int = 1,
    ) -> ModuleKernel:
        if method not in type(self)._shared_arg_indices:
            raise NotImplementedError(f"{method} is not supported")
        if isinstance(sample_inputs, torch.Tensor):
            sample_inputs = [sample_inputs]

        conv2d = self._conv2d
        parameters = {}
        parameters["in_channels"] = conv2d.in_channels
        parameters["out_channels"] = conv2d.out_channels
        parameters["kernel_size"] = conv2d.kernel_size
        parameters["stride"] = conv2d.stride
        parameters["padding"] = conv2d.padding
        parameters["dilation"] = conv2d.dilation
        parameters["groups"] = conv2d.groups

        sample_output = self.module(*sample_inputs)
        input_infos = {}
        for i, input in enumerate(sample_inputs):
            input_infos[f"input_{i}"] = list(input.shape)
        output_infos = {"output_0": list(sample_output.shape)}
        logger.debug(
            f"""
module name: {self.module_name}
input_infos: {input_infos}
output_infos: {output_infos}
parameters: {parameters}
"""
        )
        return ModuleKernel(
            module_name=self.module_name,
            method=method,
            kernel_source=None,
            input_infos=input_infos,
            output_infos=output_infos,
            parameters=parameters,
        ).load_from_db(objective_func, rank)

    def make_module(
        self,
        sample_inputs: AtomModuleInputType,
        objective_func: str = "fastest",
        rank: Union[int, List[int]] = 1,
    ) -> nn.Module:
        jit_function = self.make_function(
            sample_inputs=sample_inputs,
            mode="eval",
            objective_func=objective_func,
            rank=rank,
        )

        module_name = "BRT." + self.module_name
        extra_repr = self._conv2d.extra_repr()

        conv = self._conv2d
        if self._bn is None:
            weight = conv.weight
            bias = conv.bias
            bn_bias = None
        else:
            bn = self._bn
            var_sqrt = torch.sqrt(bn.running_var + bn.eps)
            weight = nn.Parameter(
                conv.weight
                * (bn.weight / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
            )
            if conv.bias is not None:
                bias = conv.bias
                bn_bias = bn.bias
            else:
                bias = nn.Parameter(bn.bias - bn.running_mean / var_sqrt * bn.weight)
                bn_bias = None

        return JitConv2dElemwiseModule(
            function=jit_function,
            module_name=module_name,
            extra_repr=extra_repr,
            parameters={"weight": weight, "bias": bias, "bn_bias": bn_bias},
        )

    def _extract_shared_arg_infos(self, method: str, sample_input: torch.Tensor):
        # Only support method == 'forward' actually
        if method not in type(self)._shared_arg_indices:
            raise NotImplementedError(f"{method} is not supported")
        sample_output = self.module(sample_input)
        sample_input_size = sample_input.numel() // sample_input.shape[1]
        sample_output_size = sample_output.numel() // sample_output.shape[1]
        shared_arg_grans = [
            sample_input_size * ModuleDTypeSizeInByte[sample_input.dtype],
            sample_output_size * ModuleDTypeSizeInByte[sample_output.dtype],
        ]

        return type(self)._shared_arg_indices[method], shared_arg_grans

    def _extract_arg_infos(self, method: str) -> Tuple[int, int, List[int], List[int]]:
        if method not in type(self)._supported_methods:
            raise NotImplementedError(f"{method} is not supported")
        module_name = self.module_name

        module_input_arg_num = len(self._module_inputs)
        if "Bias" in module_name:
            module_input_arg_indices = [0] + list(range(4, 3 + module_input_arg_num))
        else:
            module_input_arg_indices = [0] + list(range(3, 2 + module_input_arg_num))
        module_output_arg_indices = [2]

        function_input_arg_num = module_input_arg_num
        if "BatchNorm" in module_name:
            if "Bias" in module_name:
                function_input_arg_num += 3  # fused_weight & conv_bias & bn_bias
            else:
                function_input_arg_num += 2  # fused_weight & fused_bias
        else:
            if "Bias" in module_name:
                function_input_arg_num += 2  # weight & bias
            else:
                function_input_arg_num += 1  # weight
        kernel_input_arg_num = function_input_arg_num + 1

        return (
            function_input_arg_num,
            kernel_input_arg_num,
            module_input_arg_indices,
            module_output_arg_indices,
        )

    def _extract_parameters_and_buffers(self) -> List[Optional[torch.Tensor]]:
        conv = self._conv2d
        if self._bn is None:
            ret = [conv.weight]
            if conv.bias is not None:
                ret.append(conv.bias)
        else:
            bn = self._bn
            var_sqrt = torch.sqrt(bn.running_var + bn.eps)
            fused_weight = nn.Parameter(
                conv.weight
                * (bn.weight / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
            )
            if conv.bias is not None:
                ret = [fused_weight, conv.bias, bn.bias]
            else:
                # conv_bias = torch.zeros_like(bn.running_mean)
                fused_bias = nn.Parameter(
                    bn.bias - bn.running_mean / var_sqrt * bn.weight
                )
                ret = [fused_weight, fused_bias]
        return ret

    @property
    def module_name(self) -> str:
        module_name = "Conv2d"
        if self._conv2d.bias is not None:
            module_name += "Bias"
        for succ in self._succeed:
            if isinstance(succ, fx.Node):
                if succ.target in (operator.add, torch.add, torch.Tensor.add):
                    module_name += "Add"
                elif succ.target in (operator.mul, torch.mul, torch.Tensor.mul):
                    module_name += "Mul"
            elif isinstance(succ, type(self)._optional_succeed_module_cls):
                if isinstance(succ, torch.nn.BatchNorm2d):
                    module_name += "BatchNorm"
                else:
                    module_name += type(succ).__name__
            else:
                raise RuntimeError
        return module_name


class JitConv2dElemwiseModule(JitAtomModule):
    def __init__(
        self,
        function: Type[autograd.Function],
        module_name: str = "BRT.Conv2dElemwiseModule",
        extra_repr: str = "",
        parameters: Dict[str, torch.Tensor] = ...,
    ):
        super().__init__(function, module_name, extra_repr, parameters)
        self._factory_cls = Conv2dElementWiseModule

    def forward(self, *inputs: torch.Tensor):
        if self.bn_bias is not None:
            out = self.function.apply(
                inputs[0], self.weight, self.bias, self.bn_bias, *inputs[1:]
            )[0]
        elif self.bias is not None:
            out = self.function.apply(inputs[0], self.weight, self.bias, *inputs[1:])[0]
        else:
            out = self.function.apply(inputs[0], self.weight, *inputs[1:])[0]
        return out
