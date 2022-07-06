# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from brt.routers.protocols.protocol import ProtocolFactory

__all__ = ["make_protocol", "ProtocolFactory"]


def make_protocol(protocol_type, **kwargs):
    return ProtocolFactory.make_protocol(protocol_type, **kwargs)
