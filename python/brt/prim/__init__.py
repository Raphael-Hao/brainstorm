# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import inspect
import logging
from typing import Any, TypeVar

from .netlet import netlet
from .registry import is_netlet, is_router, is_traceable
from .router import router

LOG = logging.getLogger("brainstorm")

T = TypeVar("T")

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

def unwrap_netlet(m):
    if is_netlet(m):
        m._netlet_tag = False
        m.forward = m.pt_forward
    return m


def unwrap_redundant_netlet(m):
    if is_top_graph(m):
        LOG.debug("unwrap_redundant_netlet due to top_graph")
        unwrap_netlet(m)
    # check if router in children
    if_redundant = True
    for child in m.children():
        if is_router(child):
            if_redundant = False
            break
    if if_redundant:
        print("unwrap_redundant_netlet due to no router")
        for child in m.children():
            unwrap_netlet(child)
    for child in m.children():
        unwrap_redundant_netlet(child)
    return m


def is_top_graph(cls_or_instance) -> bool:
    if not inspect.isclass(cls_or_instance):
        cls_or_instance = cls_or_instance.__class__
    import torch
    assert issubclass(cls_or_instance, torch.nn.Module), "Only nn.Module is supported."
    return getattr(cls_or_instance, "_top_graph", False)


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
        _top_graph = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            unwrap_redundant_netlet(self)

    return wrapper
