# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
# from brt.routers.fabrics.hetero_fused import *
from brt.routers.fabrics import generic, homo_fused
from brt.routers.fabrics.fabric import FabricBase, FabricFactory

__all__ = ["FabricFactory", "make_fabric"]


def make_fabric(fabric_type, **kwargs):
    return FabricFactory.make_fabric(fabric_type, **kwargs)
