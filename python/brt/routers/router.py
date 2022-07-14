# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import inspect
from typing import Callable, Dict, List, Type, Union

import torch
import torch.nn as nn
from brt.common import log
from brt.routers.functions import convert_index_format
from brt.runtime import Registry
from brt.trace.initialize import trace_init

__all__ = ["RouterBase"]

logger = log.get_logger(__file__)


class RouterBase(nn.Module):
    def __init__(self):
        super().__init__()

    def cordinate_index_format(
        self,
        route_indices: torch.Tensor,
        loads: torch.Tensor,
        origin_index_fmt: str,
        new_index_fmt: str,
    ):
        """
        Convert the route indices to the cordinate index format.
        """
        return convert_index_format(
            route_indices, loads, origin_index_fmt, new_index_fmt
        )


def register_router(router_type: str) -> Callable:
    global_register_func = Registry.register_cls(router_type, RouterBase)

    def local_register_func(router_cls):

        router_cls = trace_init(router_cls)

        return global_register_func(router_cls)

    return local_register_func


def make_router(router_type: str, **kwargs) -> RouterBase:
    router_cls = Registry.get_cls(router_type, RouterBase)
    if router_cls is None:
        raise ValueError(f"Router type: {router_type} is not registered.")
    return router_cls(**kwargs)


def is_router(cls_or_instance) -> bool:
    if not inspect.isclass(cls_or_instance):
        router_cls = cls_or_instance.__class__
    else:
        router_cls = cls_or_instance
    return Registry.cls_exists(router_cls, RouterBase)
