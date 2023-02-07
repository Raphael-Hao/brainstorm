from typing import List, Tuple, Union, Literal, Optional

import torch
from torch import nn

from brt.jit.modules import AtomModule
from brt.jit.codegen.module import ModuleKernel


class Conv2dMulAdd(nn.Module):
    def __init__(
        self,
        conv2d,
        scale=1,
    ) -> None:
        super().__init__()
        self.conv2d = conv2d
        self.scale = scale

    def forward(self, x: torch.Tensor, add: torch.Tensor):
        x = self.conv2d(x)
        if self.scale != 1:
            x = x * self.scale
        x = x + add
        return x


# class Conv2dMulAddModule(AtomModule):

#     _input_arg_indices = {"forward": [0, 4]}
#     _output_arg_indices = {"forward": [2]}
#     _shared_arg_indices = {"forward": [0, 2, 4]}

#     def __init__(self, module: nn.Module):
#         super().__init__(module)

#     @classmethod
#     def ismodule(cls, module: torch.nn.Module):
#         if isinstance(module, Conv2dMulAdd):
#             return True
#         else:
#             return True

#     def _make_global_kernel(
#         self,
#         sample_inputs: List[torch.Tensor],
#         method: str,
#         objective_func: str = "fastest",
#         rank: int = 1,
#     ) -> ModuleKernel:
#         if method not in type(self)._shared_arg_indices:
#             raise NotImplementedError(f"{method} is not supported")
#         conv2d: nn.Conv2d = self.module.conv2d
#         parameters = {}
#         parameters["in_channels"] = conv2d.in_channels
#         parameters["out_channels"] = conv2d.out_channels
#         parameters["kernel_size"] = conv2d.kernel_size
#         parameters["stride"] = conv2d.stride
#         parameters["padding"] = conv2d.padding
#         parameters["dilation"] = conv2d.dilation
#         parameters["groups"] = conv2d.groups
#         input_infos = {
#             "input_0": list(sample_inputs[0].shape),
#             "input_1": list(sample_inputs[1].shape),
#         }
#         output_infos = {"output_0": self._get_output_shape(method, sample_inputs)[0]}
#         return ModuleKernel(
#             module_name=self.module_name,
#             method=method,
#             kernel_source=None,
#             input_infos=input_infos,
#             output_infos=output_infos,
#             parameters=parameters,
#         ).load_from_db(objective_func, rank)

#     def make_module(
#         self,
#         sample_inputs: List[torch.Tensor],
#         objective_func: str = "fastest",
#         rank: Union[int, List[int]] = 1,
#     ) -> nn.Module:
#         jit_function = self.make_function(
#             sample_inputs=sample_inputs,
#             mode="eval",
#             objective_func=objective_func,
#             rank=rank,
#         )
#         module_name = "BRT." + self.module_name
#         extra_repr = self.module.conv2d.extra_repr()

#         class JitConv2dMulAddModule(nn.Module):
#             def __init__(self, weight: torch.Tensor, bias: torch.Tensor = None) -> None:
#                 super().__init__()
#                 self._extra_repr = extra_repr
#                 self.function = jit_function
#                 self.register_parameter("weight", weight)
#                 self.register_parameter("bias", bias)
#                 self.forward(*sample_inputs)

#             def forward(self, input: torch.Tensor, add: torch.Tensor):
#                 if self.bias is not None:
#                     return self.function.apply(input, self.weight, self.bias, add)[0]
#                 else:
#                     return self.function.apply(input, self.weight, add)[0]

#             def _get_name(self):
#                 return module_name

#             def extra_repr(self):
#                 return self._extra_repr

#         return JitConv2dMulAddModule(*self._extract_parameters_and_buffers())

#     def _extract_shared_arg_infos(self, method: str, sample_input: torch.Tensor):
#         raise NotImplementedError

#     def _extract_arg_infos(self, method: str) -> Tuple[int, int, List[int], List[int]]:
#         if method not in type(self)._shared_arg_indices:
#             raise NotImplementedError(f"{method} is not supported")
#         if "Bias" not in self.module_name:
#             input_arg_num = 3
#         else:
#             input_arg_num = 4
#         total_arg_num = input_arg_num + 1
#         return (
#             input_arg_num,
#             total_arg_num,
#             type(self)._input_arg_indices[method],
#             type(self)._output_arg_indices[method],
#         )

#     def _extract_parameters_and_buffers(self) -> List[Optional[torch.Tensor]]:
#         ret = [self.module.conv2d.weight]
#         if self.module.conv2d.bias is not None:
#             ret.append(self.module.conv2d.bias)
#         return ret

#     @property
#     def module_name(self) -> str:
#         module_name = "Covn2d"
#         if self.module.conv2d.bias is not None:
#             module_name += "Bias"
#         if self.module.scale == 1:
#             module_name += "Mul"
#         module_name += "Add"
#         return module_name