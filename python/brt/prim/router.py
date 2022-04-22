# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import TypeVar

import torch

from .base import torchscript_patch, trace
from .helper import check_wrapped

T = TypeVar("T")


def router(cls: T, router_tag: bool = True) -> T:
    """Decorator for annotating the class as a router for brainstorm."""

    assert issubclass(cls, torch.nn.Module), "Only nn.Module is supported."

    if check_wrapped(cls, "router"):
        return cls

    cls = trace(cls)
    cls._brt_router = router_tag
    cls.forward = torch.jit.ignore(cls.forward)
    torchscript_patch(cls)

    return cls
