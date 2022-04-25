# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import inspect
from typing import Any, TypeVar

from brt.common import logging

from .domain import domain
from .helper import is_netlet, is_router, is_traceable
from .netlet import netlet
from .router import router

logger = logging.get_module_logger(__file__)

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


