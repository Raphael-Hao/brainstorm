# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from brt.common import log

from . import fabric, homo_fused

logger = log.get_logger(__file__)


def make_fabric(fabric_type, **kwargs):
    for key, value in kwargs.items():
        logger.debug(f"{key}: {value}")
    if fabric_type == "dispatch":
        fab = fabric.DispatchSF(**kwargs)
    elif fabric_type == "combine":
        fab = fabric.CombineSF(**kwargs)
    elif fabric_type == "homo_dispatch":
        fab = homo_fused.HomoFusedDispatchSF(**kwargs)
    elif fabric_type == "homo_combine":
        fab = homo_fused.HomoFusedCombineSF(**kwargs)
    else:
        raise ValueError
    return fab
