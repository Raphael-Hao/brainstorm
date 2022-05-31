# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import inspect
from typing import TypeVar, Union

from .helper import check_wrapped
from .serialize import torchscript_patch, trace

T = TypeVar("T")

def domain(cls: T, domain_tag=True) -> T:
    """Decorator for annotating the whole graph

    Args:
        cls (T): _description_

    Returns:
        T: _description_
    """
    if check_wrapped(cls, "domain"):
        return cls

    cls = trace(cls)

    cls._brt_domain = domain_tag
    torchscript_patch(cls)
    return cls
