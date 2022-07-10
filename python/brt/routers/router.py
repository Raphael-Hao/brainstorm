# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Callable, Dict, List, Type, Union

import torch.nn as nn
from brt.common import log
from brt.runtime import Registry

__all__ = ["RouterBase"]

logger = log.get_logger(__file__)


class RouterBase(nn.Module):
    def __init__(self, path_num: int):
        """_summary_

        Args:
            path_num (int): number of paths for routing source or destinations
            gran_dim (_type_, optional): routing granularity. should be a int or a list of int.
        """
        super().__init__()
        self.path_num = path_num


def register_router(router_type: str) -> Callable:
    return Registry.register_cls(router_type, RouterBase)


def make_router(router_type: str, **kwargs) -> RouterBase:
    router_cls = Registry.get_cls(router_type, RouterBase)
    if router_cls is None:
        raise ValueError(f"Router type: {router_type} is not registered.")
    return router_cls(**kwargs)


def is_router(cls_or_instance) -> bool:
    if isinstance(cls_or_instance, nn.Module):
        router_cls = cls_or_instance.__class__
    else:
        router_cls = cls_or_instance
    return Registry.cls_exists(router_cls, RouterBase)

