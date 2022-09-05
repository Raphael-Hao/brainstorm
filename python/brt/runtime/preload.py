# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from brt.trace.leaf_node import register_leaf_node


def pin_memory(m: nn.Module):
    for module in m.children():
        pin_memory(module)

    for key, param in m._parameters.items():
        if param is None:
            continue
        with torch.no_grad():
            param.data = param.pin_memory()
        if param.grad is not None:
            with torch.no_grad():
                param.grad.data = param.grad.pin_memory()

    for key, buf in m._buffers.items():
        if buf is not None:
            m._buffers[key] = buf.pin_memory()

    return m


@register_leaf_node
class EventEmitter(nn.Module):
    def __init__(self, event_num) -> None:
        super().__init__()
        self.event_num = event_num
        self.events = tuple(i for i in range(self.event_num))

    def forward(self, input):
        return (input,) + self.events


@register_leaf_node
class EventCollector(nn.Module):
    def __init__(self, event_num) -> None:
        super().__init__()
        self.event_num = event_num

    def forward(self, output, *events):
        return output


@register_leaf_node
class PreLoader(nn.Module):
    _stream: torch.cuda.Stream = None
    _events: List[torch.cuda.Event] = None
    _initialized: bool = False

    @classmethod
    def init(cls, event_num=1, stream=None):
        if stream is None:
            cls._stream = torch.cuda.stream()
        elif isinstance(stream, torch.cuda.Stream):
            cls._stream = stream
        else:
            cls._stream = torch.cuda.Stream()
        cls._events = [torch.cuda.Stream() for i in range(event_num)]
        cls._initialized = True

    def __init__(
        self,
        parameters: Dict[str, Parameter] = None,
        buffers: Dict[str, Tuple[torch.Tensor, nn.Module, str]] = None,
    ) -> None:
        super().__init__()
        self.collected_parameters = parameters
        self.collected_buffers = buffers
        self.event_idx = 0

    def forward(self, event_idx):
        pass

    def load_module(self, event_idx):
        m = self._load_module(m)
        PreLoader._events[event_idx].record(self._stream)
        return m

    @classmethod
    @classmethod
    def unload_module(cls, m: nn.Module):
        m = cls._unload_module(m)
        cls._event.record(cls._stream)
        return m


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
