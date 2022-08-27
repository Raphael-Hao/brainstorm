# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from brt.router.protocol.base import ProtocolBase, register_protocol


@register_protocol("identity")
class IdentityProtocol(ProtocolBase):
    def __init__(self, **kwargs):
        super().__init__(index_format=None, index_gen_opt=False, **kwargs)
    def make_route_decision(self, score, **kwargs):
        return None, None, None