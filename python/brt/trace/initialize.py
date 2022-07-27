# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import inspect
from typing import Any, TypeVar

from nni.common.serializer import is_traceable, is_wrapped_with_trace, trace

__all__ = ["netlet", "router"]

T = TypeVar("T")


def get_init_arguments(obj: Any):
    if is_traceable(obj):
        return obj.trace_kwargs
    raise ValueError(
        f"Object {obj} needs to be serializable but `trace_kwargs` is not available. "
        "If it is a built-in module (like Conv2d) or nn.Module composed of torch's built-in modules, "
        "please decorate it with brt.netlet."
    )


def trace_init(cls, traced_type="router"):
    """
    Decorator for annotating an nn.Module as a Netlet.
    """
    assert traced_type in ["netlet", "router"]
    if check_wrapped(cls, "netlet") or check_wrapped(cls, "router"):
        return cls

    cls = trace(cls)

    tag = "_brt_" + traced_type

    setattr(cls, tag, True)

    _torchscript_patch(cls)
    return cls


def is_wrapped_router(cls_or_instance) -> bool:
    """check if the class is a router for brainstorm."""
    if not inspect.isclass(cls_or_instance):
        cls_or_instance = cls_or_instance.__class__

    import torch

    assert issubclass(
        cls_or_instance, torch.nn.Module
    ), "Only nn.Module is supported currently."
    return getattr(cls_or_instance, "_brt_router", False)


def is_wrapped_netlet(cls_or_instance) -> bool:
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
    if is_wrapped_netlet(cls):
        wrapped = "netlet"
    elif is_wrapped_netlet(cls):
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


def _torchscript_patch(cls) -> None:
    # HACK: for torch script
    # https://github.com/pytorch/pytorch/pull/45261
    # https://github.com/pytorch/pytorch/issues/54688
    # I'm not sure whether there will be potential issues
    import torch

    if hasattr(cls, "_get_brt_attr"):  # could not exist on non-linux
        cls._get_brt_attr = torch.jit.ignore(cls._get_brt_attr)
    if hasattr(cls, "trace_symbol"):
        # these must all exist or all non-exist
        try:
            cls.trace_symbol = torch.jit.unused(cls.trace_symbol)
            cls.trace_args = torch.jit.unused(cls.trace_args)
            cls.trace_kwargs = torch.jit.unused(cls.trace_kwargs)
            cls.trace_copy = torch.jit.ignore(cls.trace_copy)
        except AttributeError as e:
            if "property" in str(e):
                raise RuntimeError(
                    "Trace on PyTorch module failed. Your PyTorch version might be outdated. "
                    "Please try to upgrade PyTorch."
                )
            raise
