# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import TypeVar
import inspect

import torch

__all__ = ["is_top_graph", "is_netlet", "is_router", "top_graph", "netlet"]
T = TypeVar("T")


def is_top_graph(cls_or_instance) -> bool:
    if not inspect.isclass(cls_or_instance):
        cls_or_instance = cls_or_instance.__class__

    assert issubclass(cls_or_instance, torch.nn.Module), "Only nn.Module is supported."
    return getattr(cls_or_instance, "_top_graph", False)


def is_netlet(cls_or_instance) -> bool:
    """
    Check if the class is a netlet for brainstorm.
    """
    if not inspect.isclass(cls_or_instance):
        cls_or_instance = cls_or_instance.__class__

    assert issubclass(cls_or_instance, torch.nn.Module), "Only nn.Module is supported."
    return getattr(cls_or_instance, "_netlet", False)


def is_router(cls_or_instance) -> bool:
    """check if the class is a router for brainstorm."""
    if not inspect.isclass(cls_or_instance):
        cls_or_instance = cls_or_instance.__class__

    assert issubclass(cls_or_instance, torch.nn.Module), "Only nn.Module is supported."
    return getattr(cls_or_instance, "_router_tag", False)


def top_graph(
    cls: T,
) -> T:
    """Decorator for annotating the whole graph

    Args:
        cls (T): _description_

    Returns:
        T: _description_
    """

    class wrapper(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def brt_script(self, mode: bool = True):
            def switch_mode(m):
                if getattr(m, "_netlet", False):
                    m.brt_script(mode)

            self.apply(switch_mode)

    wrapper._top_graph = True
    return wrapper


def netlet(cls: T, netlet_tag: bool = True) -> T:
    """
    Decorator for annotating an nn.Module as a Netlet.
    TODO check why we don't need to switch forward method
    TODO can we use torch.jit.is_scripting for a more robust implementation?
    """
    if is_netlet(cls):
        return cls
    forward_sig = inspect.signature(cls.forward)

    class wrapper(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._brt_scripting = False
            self.pt_forward = super().forward
            self.forward = self.brt_forward

        @torch.jit.ignore
        def brt_forward(self, *args, **kwargs):
            narg = len(args) + len(kwargs)
            if narg == 1 and args[0] == None:
                return None
            return self.pt_forward(*args, **kwargs)

        # def forward(self, *args, **kwargs):
        #     if torch.jit.is_scripting():
        #         return self.pt_forward(*args, **kwargs)
        #     else:
        #         return self.brt_forward(*args, **kwargs)

        # def brt_script(self, mode: bool = True):
        #     self._brt_scripting = mode
        #     if self._brt_scripting:
        #         self.forward = self.pt_forward
        #     else:
        #         self.forward = self.brt_forward

    wrapper.brt_forward.__signature__ = forward_sig
    wrapper._netlet_tag = netlet_tag

    return wrapper
