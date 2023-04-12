# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import inspect
from typing import Callable, Dict, Any

import torch.nn as nn

from brt.runtime import log, Registry
from brt.router.utils import make_kwargs
from brt.trace.init_arg import trace_init

__all__ = ["RouterBase"]

logger = log.get_logger(__file__)


class RouterBase(nn.Module):
    def __init__(self):
        super().__init__()

def register_router(router_type: str) -> Callable:
    global_register_func = Registry.register_sub_cls(router_type, RouterBase)

    def local_register_func(router_cls):

        if not issubclass(router_cls, RouterBase):
            raise ValueError(f"{router_cls} is not a subclass of RouterBase")

        router_cls = trace_init(router_cls)

        router_cls._router_type = router_type

        return global_register_func(router_cls)

    return local_register_func


def make_router(router_type: str, kwargs: Dict[str, Any]) -> RouterBase:
    router_cls = Registry.get_sub_cls(router_type, RouterBase)
    if router_cls is None:
        raise ValueError(f"Router type: {router_type} is not registered.")
    formulated_kwargs = make_kwargs(kwargs)
    return router_cls(**formulated_kwargs)


def is_router(cls_or_instance) -> bool:

    if not inspect.isclass(cls_or_instance):
        router_cls = cls_or_instance.__class__
    else:
        router_cls = cls_or_instance

    return Registry.sub_cls_exists_and_registered(router_cls, RouterBase)



