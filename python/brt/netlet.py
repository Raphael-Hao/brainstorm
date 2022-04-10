# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, TypeVar
import inspect

import torch

__all__ = ["is_netlet", "netlet", "branch"]
T = TypeVar("T")


def is_netlet(cls_or_instance) -> bool:
    """
    Check if the class is a subclass of nn.Module.
    """
    if not inspect.isclass(cls_or_instance):
        cls_or_instance = cls_or_instance.__class__

    import torch.nn as nn

    assert issubclass(cls_or_instance, nn.Module), "Only nn.Module is supported."
    return getattr(cls_or_instance, "brt_netlet", False)


def branch(func: T) -> T:
    """
    Decorator for annotating a forward function as a branch.
    """

    @torch.jit.ignore
    def branch_forward(*args: Any, **kwargs: Any) -> Any:
        narg = len(args) + len(kwargs)
        if narg == 1 and args[0] == None:
            return None
        return func(*args, **kwargs)

    return branch_forward


def brt_script(self, mode: bool = True):
    self._brt_script = mode
    self.apply(lambda m: m.brt_script(mode) if hasattr(m, "_brt_netlet") else None)


def brt_patch(cls: T) -> T:
    brt_pt_forward = cls.forward
    brt_branch_forward = branch(brt_pt_forward)
    setattr(cls, "brt_pt_forward", brt_pt_forward)
    setattr(cls, "brt_branch_forward", brt_branch_forward)
    setattr(cls, "brt_script", brt_script)
    cls.forward = cls.brt_branch_forward
    return cls


def netlet(cls: T, netlet_tag: bool = True) -> T:
    """
    Decorator for annotating an nn.Module as a Netlet.
    """
    if is_netlet(cls):
        return cls
    class wrapper(cls):
        def __init__(self, *args, **kwargs):
            self._brt_script = False
            super().__init__(*args, **kwargs)
            self.pt_forward = super().forward
            self.forward = self.brt_forward

        @torch.jit.ignore
        def brt_forward(self, *args, **kwargs):
            narg = len(args) + len(kwargs)
            if narg == 1 and args[0] == None:
                return None
            return self.pt_forward(*args, **kwargs)

        def brt_script(self, mode: bool = True):
            for module in self.children():
                if getattr(module, "_netlet", False):
                    module.brt_script(mode)
            self._brt_script = mode
            if self._brt_script:
                self.forward = self.pt_forward
            else:
                self.forward = self.brt_forward
            # self.apply(lambda m: m.brt_script(mode) if hasattr(m, "_netlet") else None)

    wrapper._netlet = netlet_tag

    return wrapper
