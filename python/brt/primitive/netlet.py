# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import TypeVar

from brt.common import log

from .helper import check_wrapped, is_wrapped_with_trace
from .serialize import torchscript_patch, trace

logger = log.get_logger(__file__)

__all__ = ["netlet"]
T = TypeVar("T")


def netlet(cls: T, netlet_tag: bool = True) -> T:
    """
    Decorator for annotating an nn.Module as a Netlet.
    TODO check why we don't need to switch forward method
    """
    if check_wrapped(cls, "netlet"):
        return cls

    cls = trace(cls)

    cls._brt_netlet = netlet_tag
    torchscript_patch(cls)
    return cls
