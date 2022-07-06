# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from .fabric import FabricFactory

def make_fabric(fabric_type, **kwargs):
    FabricFactory.make_fabric(fabric_type, **kwargs)
    return fabric