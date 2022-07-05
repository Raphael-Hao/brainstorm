# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import torch
from brt.common import log

from . import fabric, homo_fused

logger = log.get_logger(__file__)


def make_fabric(fabric_type, **kwargs):
    for key, value in kwargs.items():
        logger.debug(f"{key}: {value}")
    if fabric_type == "dispatch":
        fabric_cls = fabric.DispatchSF
    elif fabric_type == "combine":
        fabric_cls = fabric.CombineSF
    elif fabric_type == "homo_dispatch":
        fabric_cls = homo_fused.HomoFusedDispatchSF
    elif fabric_type == "homo_combine":
        fabric_cls = homo_fused.HomoFusedCombineSF
    else:
        raise ValueError(f"Unknown fabric type: {fabric_type}")

    fabric_cls.forward = torch.jit.ignore(fabric_cls.forward)
    fab = fabric_cls(**kwargs)
    return fab
