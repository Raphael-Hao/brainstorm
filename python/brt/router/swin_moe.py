# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from brt.router.generic import ScatterRouter, GatherRouter
from brt.router.base import register_router, RouterBase


@register_router("swin_moe_scatter")
class SwinMoEScatterRouter(RouterBase):
    def __init__(
        self,
        protocol_type: str = "swin_moe",
        fabric_type: str = "dispatch",
        
    ):
        super().__init__()
