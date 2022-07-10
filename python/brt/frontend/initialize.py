# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import inspect
from typing import TypeVar

from nni.retiarii.serializer
import torch

from .serialize import is_wrapped_with_trace, torchscript_patch, trace

__all__ = ["netlet", "router", "symbolize", "de_symbolize"]

T = TypeVar("T")


def trace_init(cls, traced_type="router"):
    """
    Decorator for annotating an nn.Module as a Netlet.
    """
    assert traced_type in ["netlet", "router"]
    if check_wrapped(cls, "netlet") or check_wrapped(cls, "router"):
        return cls

    cls = trace(cls)

    tag = "brt_" + traced_type

    setattr(cls, tag, True)

    torchscript_patch(cls)
    return cls


def is_router(cls_or_instance) -> bool:
    """check if the class is a router for brainstorm."""
    if not inspect.isclass(cls_or_instance):
        cls_or_instance = cls_or_instance.__class__
    import torch

    assert issubclass(
        cls_or_instance, torch.nn.Module
    ), "Only nn.Module is supported for router."
    return getattr(cls_or_instance, "_brt_router", False)


def is_netlet(cls_or_instance) -> bool:
    """
    Check if the class is a netlet for brainstorm.
    """
    if not inspect.isclass(cls_or_instance):
        cls_or_instance = cls_or_instance.__class__

    import torch

    assert issubclass(
        cls_or_instance, torch.nn.Module
    ), "Only nn.Module is supported currently."
    return getattr(cls_or_instance, "_brt_netlet", False)


def check_wrapped(cls: T, rewrap: str) -> bool:
    wrapped = None
    if is_netlet(cls):
        wrapped = "netlet"
    elif is_router(cls):
        wrapped = "router"
    elif is_wrapped_with_trace(cls):
        wrapped = "trace"
    if wrapped:
        if wrapped != rewrap:
            raise TypeError(
                f"{cls} is already wrapped with {wrapped}. Cannot rewrap with {rewrap}."
            )
        return True
    return False


def netlet(cls: T, netlet_tag: bool = True) -> T:
    """
    Decorator for annotating an nn.Module as a Netlet.
    """
    if check_wrapped(cls, "netlet"):
        return cls

    cls = trace(cls)

    cls._brt_netlet = netlet_tag
    torchscript_patch(cls)
    return cls


def router(cls: T, router_tag: bool = True) -> T:
    """Decorator for annotating the class as a router for brainstorm."""

    assert issubclass(cls, torch.nn.Module), "Only nn.Module is supported."

    if check_wrapped(cls, "router"):
        return cls

    cls = trace(cls)
    cls._brt_router = router_tag
    torchscript_patch(cls)

    return cls
