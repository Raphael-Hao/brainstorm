# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from brt.routers.protocols import generic, swin_moe
from brt.routers.protocols.protocol import ProtocolBase, ProtocolFactory

__all__ = ["make_protocol", "ProtocolFactory", "ProtocolBase"]

def make_protocol(protocol_type, **kwargs):
    return ProtocolFactory.make_protocol(protocol_type, **kwargs)
