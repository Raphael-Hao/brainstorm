# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from brt.routers.fabrics.fabric import FabricFactory
from brt.routers.fabrics.generic import *

# from brt.routers.fabrics.hetero_fused import *
from brt.routers.fabrics.homo_fused import *

__all__ = ["FabricFactory", "make_fabric"]


def make_fabric(fabric_type, **kwargs):
    return FabricFactory.make_fabric(fabric_type, **kwargs)
