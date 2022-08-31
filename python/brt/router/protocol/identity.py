# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Dict, Any

from brt.router.utils import assert_compatibility
from brt.router.protocol.base import ProtocolBase, register_protocol


@register_protocol("identity")
class IdentityProtocol(ProtocolBase):
    def __init__(self, **kwargs):
        self.check_compatibility(kwargs)
        super().__init__(index_format=None, index_gen_opt=False, **kwargs)

    def check_compatibility(self, kwargs: Dict[str, Any]) -> None:
        index_format = kwargs.pop("index_format", None)
        # assert_compatibility(self, "index_format", None, index_format)
        index_gen_opt = kwargs.pop("index_gen_opt", False)
        # assert_compatibility(self, "index_gen_opt", False, index_gen_opt)

    def make_route_decision(self, score, **kwargs):
        return None, None, None
