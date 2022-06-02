# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import inspect
from typing import Any, TypeVar

T = TypeVar("T")


def is_traceable(obj: Any) -> bool:
    """
    Check whether an object is a traceable instance or type.

    Note that an object is traceable only means that it implements the "Traceable" interface,
    and the properties have been implemented. It doesn't necessary mean that its type is wrapped with trace,
    because the properties could be added **after** the instance has been created.
    """
    return (
        hasattr(obj, "trace_copy")
        and hasattr(obj, "trace_symbol")
        and hasattr(obj, "trace_args")
        and hasattr(obj, "trace_kwargs")
    )


def is_wrapped_with_trace(cls_or_func: Any) -> bool:
    """
    Check whether a function or class is already wrapped with ``@brt.trace``.
    If a class or function is already wrapped with trace, then the created object must be "traceable".
    """
    return getattr(cls_or_func, "_traced", False) and (
        not hasattr(cls_or_func, "__dict__")
        or "_traced"  # in case it's a function
        in cls_or_func.__dict__  # must be in this class, super-class traced doesn't count
    )


def is_domain(cls_or_instance) -> bool:
    """check if the class is a domain for brainstorm."""

    if not inspect.isclass(cls_or_instance):
        cls_or_instance = cls_or_instance.__class__
    import torch
    assert issubclass(cls_or_instance, torch.nn.Module), "Only nn.Module is supported."
    return getattr(cls_or_instance, "_brt_domain", False)

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

def _switch_symbolic(m, symbolic = True):
    for child in m.children():
        _switch_symbolic(child, symbolic)
    if is_router(m):
        m._brt_symbolic = symbolic
        if m._brt_symbolic:
            m.forward = m.symbolic_route
        else:
            m.forward = m.route
    return m

def symbolize(m):
    return _switch_symbolic(m)

def de_symbolize(m):
    return _switch_symbolic(m, False)

def get_init_parameters_or_fail(obj: Any):
    if is_traceable(obj):
        return obj.trace_kwargs
    raise ValueError(
        f"Object {obj} needs to be serializable but `trace_kwargs` is not available. "
        "If it is a built-in module (like Conv2d), please import it from retiarii.nn. "
        "If it is a customized module, please to decorate it with @basic_unit. "
        "For other complex objects (e.g., trainer, optimizer, dataset, dataloader), "
        "try to use @nni.trace."
    )