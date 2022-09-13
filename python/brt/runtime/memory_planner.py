# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Dict, Tuple
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from brt.trace.leaf_node import register_leaf_node
from brt.runtime import log

logger = log.get_logger(__file__)

__all__ = [
    "pin_memory",
    "OnDemandLoader",
    "OnDemandGuarder",
    "OnDemandUnloader",
    "PredictLoader",
    "PredictGuarder",
    "PredictUnloader",
]


def pin_memory(m: nn.Module):
    for module in m.children():
        pin_memory(module)

    for key, param in m._parameters.items():
        if param is None:
            continue
        with torch.no_grad():
            param.data = param.pin_memory()
        param.pin_cpu_data = param.data
        if param.grad is not None:
            with torch.no_grad():
                param.grad.data = param.grad.pin_memory()
            param.grad.pin_cpu_data = param.grad.data

    for key, buf in m._buffers.items():
        if buf is not None:
            m._buffers[key] = buf.pin_memory()
            m._buffers[key].pin_cpu_data = m._buffers[key]

    return m


def _get_target_input(inputs, path_id):
    if isinstance(inputs, (tuple, list)):
        if len(inputs) > path_id and isinstance(inputs[path_id], torch.Tensor):
            logger.debug(
                f"scatter outputs: {list(inputs[path_id].shape)}, {inputs[path_id].device}, path id: {path_id}"
            )
            return inputs[path_id]
        else:
            return _get_target_input(inputs[0], path_id)

    if isinstance(inputs, torch.Tensor) and path_id == 0:
        logger.debug(f"gather outputs: {list(inputs.shape)}, {inputs.device}")
        return inputs

    raise ValueError("Invalid input type: {}".format(type(inputs)))


class MemoryPlanContext:
    MEMORY_STREAM: torch.cuda.Stream = None
    COMPUTE_STREAM: torch.cuda.Stream = None
    EVENTS: List[torch.cuda.Event] = None
    INITIALIZED: bool = False

    @classmethod
    def init(cls, event_num=1, memory_stream=None, compute_stream=None):
        if isinstance(memory_stream, torch.cuda.Stream):
            cls.MEMORY_STREAM = memory_stream
        else:
            cls.MEMORY_STREAM = torch.cuda.Stream()
        if isinstance(compute_stream, torch.cuda.Stream):
            cls.COMPUTE_STREAM = compute_stream
        else:
            cls.COMPUTE_STREAM = torch.cuda.current_stream()

        cls.EVENTS = [torch.cuda.Event() for _ in range(event_num)]
        cls.INITIALIZED = True

    @classmethod
    def set_memory_stream(cls, stream: torch.cuda.Stream):
        cls.MEMORY_STREAM = stream

    @classmethod
    def set_compute_stream(cls, stream: torch.cuda.Stream):
        cls.COMPUTE_STREAM = stream

class MemoryPlanner(nn.Module):
    def __init__(
        self,
        event_id: int,
        collected_params: Dict[str, Parameter] = None,
        collected_buffers: Dict[str, Tuple[torch.Tensor, nn.Module, str]] = None,
    ) -> None:
        super().__init__()
        assert (
            MemoryPlanContext.INITIALIZED
        ), "MemPlanContext is not initialized before creating a PreLoader instance"
        self.event_id = event_id
        self.collected_params = collected_params
        self.collected_buffers = collected_buffers

    def load(self, event_id: int):
        with torch.cuda.stream(MemoryPlanContext.MEMORY_STREAM):
            for pname, param in self.collected_params.items():
                logger.debug(f"load param: {pname}")
                if pname is None:
                    continue
                with torch.no_grad():
                    # print(f"before load param device: {param.device}")
                    param_applied = param.cuda(non_blocking=True)
                param.pin_cpu_data = param.data
                param.data = param_applied
                out_param = param
                # print(f"after load param device: {param.device}")

                if param.grad is not None:
                    with torch.no_grad():
                        grad_applied = param.grad.cuda(non_blocking=True)

                    out_param.grad.pin_cpu_data = param.grad.data
                    out_param.grad.data = grad_applied

            for bname, (
                buf,
                b_owner_m,
                b_tensor_name,
            ) in self.collected_buffers.items():
                # print(f"load buffer: {bname}")
                if buf is not None:
                    b_owner_m._buffers[b_tensor_name] = buf.cuda(non_blocking=True)

        MemoryPlanContext.EVENTS[event_id].record(MemoryPlanContext.MEMORY_STREAM)

    def guard(self, event_id: int):
        MemoryPlanContext.EVENTS[event_id].wait(MemoryPlanContext.COMPUTE_STREAM)

    def unload(self, event_id: int):
        for pname, param in self.collected_params.items():
            if param is None:
                continue
            with torch.no_grad():
                # print(f"before unload param device: {param.device}")
                param_applied = param.pin_cpu_data
            cuda_param = param.data
            param.data = param_applied
            del cuda_param
            out_param = param
            # print(f"after unload param device: {param.device}")

            if param.grad is not None:
                with torch.no_grad():
                    # print(f"before unload param.grad device: {param.grad.device}")
                    grad_applied = param.grad.pin_cpu_data
                cuda_grad = out_param.grad.data
                out_param.grad.data = grad_applied
                del cuda_grad
                # print(f"After unload param.grad device: {out_param.grad.device}")

        for bname, (buf, b_owner_m, b_tensor_name) in self.collected_buffers.items():
            if buf is not None:
                cuda_buffer = b_owner_m._buffers[b_tensor_name]
                b_owner_m._buffers[b_tensor_name] = buf
                del cuda_buffer


class InitialLoader(MemoryPlanner):
    def __init__(self, collected_params, collected_buffers):
        super().__init__(0, collected_params, collected_buffers)

    def forward(self):
        self.load(self.event_id)
        self.guard(self.event_id)


@register_leaf_node
class OnDemandLoader(MemoryPlanner):
    def __init__(
        self,
        path_id: int,
        event_id: int,
        collected_params: Dict[str, Parameter] = None,
        collected_buffers: Dict[str, Tuple[torch.Tensor, nn.Module, str]] = None,
    ) -> None:
        super().__init__(event_id, collected_params, collected_buffers)
        self.path_id = path_id

    def forward(self, inputs):
        target = _get_target_input(inputs, self.path_id)
        if target.numel() > 0:
            self.load(self.event_id)

        return inputs


@register_leaf_node
class OnDemandGuarder(MemoryPlanner):
    def __init__(self, path_id: int, event_id: int) -> None:
        super().__init__(event_id)
        self.path_id = path_id

    def forward(self, inputs):
        target = _get_target_input(inputs, self.path_id)
        if target.numel() > 0:
            self.guard(self.event_id)
        return inputs


@register_leaf_node
class OnDemandUnloader(MemoryPlanner):
    def __init__(
        self,
        event_id: int,
        collected_params: Dict[str, Parameter] = None,
        collected_buffers: Dict[str, Tuple[torch.Tensor, nn.Module, str]] = None,
    ) -> None:
        super().__init__(event_id, collected_params, collected_buffers)

    def forward(self, inputs):
        self.unload(self.event_id)
        return inputs


@register_leaf_node
class PredictLoader(MemoryPlanner):
    def __init__(
        self,
        event_id: int,
        collected_params: Dict[str, Parameter] = None,
        collected_buffers: Dict[str, Tuple[torch.Tensor, nn.Module, str]] = None,
    ) -> None:
        super().__init__(event_id, collected_params, collected_buffers)

    def forward(self, inupts):
        self.load(self.event_id)
        return inupts


@register_leaf_node
class PredictGuarder(MemoryPlanner):
    def __init__(self, event_id: int) -> None:
        super().__init__(event_id)

    def forward(self, inputs):
        self.guard(self.event_id)
        return inputs


@register_leaf_node
class PredictUnloader(MemoryPlanner):
    def __init__(
        self,
        event_id: int,
        collected_params: Dict[str, Parameter] = None,
        collected_buffers: Dict[str, Tuple[torch.Tensor, nn.Module, str]] = None,
    ) -> None:
        super().__init__(event_id, collected_params, collected_buffers)

    def forward(self, inputs):
        self.unload(self.event_id)
        return inputs


def load_module(m: nn.Module):
    for module in m.children():
        load_module(module)

    for key, param in m._parameters.items():
        if param is None:
            continue
        # Tensors stored in modules are graph leaves, and we don't want to
        # track autograd history of `param_applied`, so we have to use
        # `with torch.no_grad():`
        with torch.no_grad():
            with torch.cuda.stream(torch.cuda.current_stream()):
                param_applied = param.cuda(non_blocking=True)
        param.pin_cpu_data = param.data
        param.data = param_applied
        out_param = param

        if param.grad is not None:
            with torch.no_grad():
                with torch.cuda.stream(torch.cuda.current_stream()):
                    grad_applied = param.grad.cuda(non_blocking=True)

            out_param.grad.pin_cpu_data = param.grad.data
            out_param.grad.data = grad_applied

    for key, buf in m._buffers.items():
        if buf is not None:
            with torch.cuda.stream(torch.cuda.current_stream()):
                m._buffers[key] = buf.cuda(non_blocking=True)
            m._buffers[key].pin_cpu_data = buf
    return m


def unload_module(m: nn.Module, copy=False):
    for module in m.children():
        unload_module(module, copy)

    for key, param in m._parameters.items():
        if param is None:
            continue
        # Tensors stored in modules are graph leaves, and we don't want to
        # track autograd history of `param_applied`, so we have to use
        # `with torch.no_grad():`
        with torch.no_grad():
            with torch.cuda.stream(torch.cuda.current_stream()):
                param_applied = param.pin_cpu_data
                if copy:
                    param_applied.copy_(param.data, non_blocking=True)
        param.data = param_applied
        out_param = param

        if param.grad is not None:
            with torch.no_grad():
                with torch.cuda.stream(torch.cuda.current_stream()):
                    grad_applied = param.grad.pin_cpu_data
                    if copy:
                        grad_applied.copy_(param.grad.data, non_blocking=True)
            out_param.grad.data = grad_applied

    for key, buf in m._buffers.items():
        if buf is not None:
            with torch.cuda.stream(torch.cuda.current_stream()):
                buf_applied = buf.pin_cpu_data
                if copy:
                    buf_applied.copy_(buf, non_blocking=True)
            m._buffers[key] = buf_applied
    return m
