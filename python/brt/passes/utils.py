# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from brt.router import RouterBase

__all__ = ["is_scatter", "is_gather"]

def is_scatter(m):
    if isinstance(m, RouterBase) and "scatter" in m._router_type:
        return True
    return False

def is_gather(m):
    if isinstance(m, RouterBase) and "gather" in m._router_type:
        return True
    return False

