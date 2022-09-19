# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from __future__ import annotations
from typing import Dict, Tuple, List

import torch
import torch.nn as nn


__all__ = ["group_params_buffers"]

class TensorGroupManager:
    """Manage a group of tensors."""

    tensor_groups: List[TensorGroup] = []

    @classmethod
    def acquire_tensor_group(cls, size_in_byte: int, target: None):
        """Get a tensor group by name."""
        for tensor_group in cls.tensor_groups:
            if tensor_group.acquire(size_in_byte, target):
                return tensor_group
        return cls._new_tensor_group(size_in_byte, target)

    @classmethod
    def _new_tensor_group(cls, size_in_byte: int, target: None):
        """Create a new tensor group."""
        tensor_group = TensorGroup(size_in_byte, target)
        cls.tensor_groups.append(tensor_group)
        return tensor_group

    @classmethod
    def release_tensor_group(cls, tensor_group: TensorGroup):
        """Release a tensor group."""
        tensor_group.release()

class TensorGroup:
    def __init__(self, size_in_byte: int, target=None) -> None:
        self.size_in_byte = size_in_byte
        self.target = "cuda" if target is None else target
        self.cpu_pin_tensor = torch.empty(
            self.size_in_byte, dtype=torch.uint8
        ).pin_memory()
        self.cpu_pin_storage = self.cpu_pin_tensor.storage()._untyped()
        self.targe_tensor = torch.empty_like(self.cpu_pin_tensor, device=self.target)
        self.targe_storage = self.targe_tensor.storage()._untyped()
        self.used_in_byte = 0
        self.use_count = 0

    def acquire(self, size_in_byte: int, target=None):
        if self.size_in_byte < size_in_byte:
            return False
        if target is not None and self.target != target:
            return False
        if self.use_count >= 1:
            return False
        self.use_count += 1
        self.used_in_byte = 0
        return True

    def release(self):
        self.use_count -= 1

    def include_tensor(self, t: torch.Tensor):
        t_size_in_byte = t.element_size() * t.numel()
        with torch.no_grad():
            t_data_pin = torch.tensor([], dtype=t.dtype, device="cpu")
            t_data_target = torch.tensor([], dtype=t.dtype, device=self.target)
            t_data_pin.set_(
                self.cpu_pin_storage,
                int(self.used_in_byte / t_data_pin.element_size()),
                t.size(),
                t.stride(),
            )
            t_data_pin.copy_(t)
            t_data_target.set_(
                self.target_storage,
                int(self.targe_storage / t_data_target.element_size()),
                t.size(),
                t.stride(),
            )
            t.data = t_data_target
        self.used_in_byte += t_size_in_byte
        assert self.used_in_byte <= self.size_in_byte, "TensorGroup is full"

    def load(self, stream):
        with torch.cuda.stream(stream):
            with torch.no_grad():
                self.targe_tensor.copy_(self.cpu_pin_tensor, non_blocking=True)

    def unload(self, stream, copy_back=True):
        if copy_back:
            with torch.cuda.stream(stream):
                with torch.no_grad():
                    self.cpu_pin_tensor.copy_(self.targe_tensor, non_blocking=True)


