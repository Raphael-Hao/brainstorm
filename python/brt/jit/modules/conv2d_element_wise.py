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
        node: fx.Node
        for node in self.module.graph.nodes:
            if node.op == "placeholder":
                self._module_inputs.append(node.name)
            elif node.op == "get_attr":
                pass
            elif node.op == "call_module":
                if (
                    isinstance(self.module.get_submodule(node.target), nn.Conv2d)
                    and node.args[0].name in self._module_inputs
                ):
                    self._conv2d = self.module.get_submodule(node.target)
                    self._conv2d_inputs = node.args[0].name
                    conv2d_output = node.name
                elif isinstance(
                    self.module.get_submodule(node.target),
                    type(self)._optional_succeed_module_cls,
                ):
                    self._succeed.append(self.module.get_submodule(node.target))
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
                    return False
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
        return JitConv2dElemwiseModule(
            function=jit_function,
            module_name=module_name,
            extra_repr=extra_repr,
            parameters={"weight": self._conv2d.weight, "bias": self._conv2d.bias,},
        )

    def _extract_shared_arg_infos(self, method: str, sample_input: torch.Tensor):
        # Only support method == 'forward' actually
        if method not in type(self)._shared_arg_indices:
            raise NotImplementedError(f"{method} is not supported")
        sample_output = self.module(sample_input)
        sample_input_size = sample_input.numel() / sample_input.shape[1]
        sample_output_size = sample_output.numel() / sample_output.shape[1]
        shared_arg_grans = [
            sample_input_size * ModuleDTypeSizeInByte[sample_input.dtype],
            sample_output_size * ModuleDTypeSizeInByte[sample_output.dtype],
        ]

        return type(self)._shared_arg_indices[method], shared_arg_grans

    def _extract_arg_infos(self, method: str) -> Tuple[int, int, List[int], List[int]]:
        if method not in type(self)._supported_methods:
            raise NotImplementedError(f"{method} is not supported")
        input_arg_num = len(self._module_inputs)
        if "Bias" in self.module_name:
            input_arg_indices = [0] + list(range(4, 3 + input_arg_num))
            input_arg_num += 2  # weight & bias
        else:
            input_arg_indices = [0] + list(range(3, 2 + input_arg_num))
            input_arg_num += 1  # weight
        total_arg_num = input_arg_num + 1
        output_arg_indices = [2]
        return (
            input_arg_num,
            total_arg_num,
            input_arg_indices,
            output_arg_indices,
        )

    def _extract_parameters_and_buffers(self) -> List[Optional[torch.Tensor]]:
        ret = [self._conv2d.weight]
        if self._conv2d.bias is not None:
            ret.append(self._conv2d.bias)
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
        function: autograd.Function,
        module_name: str = "BRT.Conv2dElemwiseModule",
        extra_repr: str = "",
        parameters: Dict[str, torch.Tensor] = ...,
    ):
        super().__init__(function, module_name, extra_repr, parameters)
        self._factory_cls = Conv2dElementWiseModule

    def forward(self, *inputs: torch.Tensor):
        if self.bias is not None:
            out = self.function.apply(inputs[0], self.weight, self.bias, *inputs[1:])[0]
        else:
            out = self.function.apply(inputs[0], self.weight, *inputs[1:])[0]
        return out

