# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class WeightLoader:
    _stream: torch.cuda.Stream = None

    @classmethod
    def init(cls, stream=None):
        if stream is None:
            cls._stream = torch.cuda.default_stream()
        else:
            cls._stream = stream

    @classmethod
    def pin_memory(cls, m: nn.Module):
        for module in m.children():
            cls.pin_memory(module)

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

    @classmethod
    def load(cls, m: nn.Module):
        for module in m.children():
            cls.load(module)

        for key, param in m._parameters.items():
            if param is None:
                continue
            # Tensors stored in modules are graph leaves, and we don't want to
            # track autograd history of `param_applied`, so we have to use
            # `with torch.no_grad():`
            with torch.no_grad():
                with torch.cuda.stream(cls._stream):
                    param_applied = param.cuda(non_blocking=True)
            param.pin_cpu_data = param.data
            param.data = param_applied
            out_param = param

            if param.grad is not None:
                with torch.no_grad():
                    with torch.cuda.stream(cls._stream):
                        grad_applied = param.grad.cuda(non_blocking=True)

                out_param.grad.pin_cpu_data = param.grad.data
                out_param.grad.data = grad_applied

        for key, buf in m._buffers.items():
            if buf is not None:
                with torch.cuda.stream(cls._stream):
                    m._buffers[key] = buf.cuda(non_blocking=True)
                m._buffers[key].pin_cpu_data = buf

        return m

    @classmethod
    def unload(cls, m: nn.Module):
        for module in m.children():
            cls.unload(module)

        for key, param in m._parameters.items():
            if param is None:
                continue
            # Tensors stored in modules are graph leaves, and we don't want to
            # track autograd history of `param_applied`, so we have to use
            # `with torch.no_grad():`
            with torch.no_grad():
                with torch.cuda.stream(cls._stream):
                    param_applied = param.pin_cpu_data
                    param_applied.copy_(param.data, non_blocking=True)
            param.data = param_applied
            out_param = param

            if param.grad is not None:
                with torch.no_grad():
                    with torch.cuda.stream(cls._stream):
                        grad_applied = param.grad.pin_cpu_data
                        grad_applied.copy_(param.grad.data, non_blocking=True)
                out_param.grad.data = grad_applied

        for key, buf in m._buffers.items():
            if buf is not None:
                with torch.cuda.stream(cls._stream):
                    buf_applied = buf.pin_cpu_data
                    buf_applied.copy_(buf, non_blocking=True)
                m._buffers[key] = buf_applied

        return m
