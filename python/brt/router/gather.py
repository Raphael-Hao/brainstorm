# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, Any

from brt.runtime import log
from brt.router.base import RouterBase, register_router
from brt.router.fabric import make_fabric

# from brt.router.utils import empty_flows

__all__ = [
    "GatherRouter",
]

logger = log.get_logger(__file__)


@register_router("gather")
class GatherRouter(RouterBase):
    def __init__(
        self,
        fabric_type: str = "combine",
        fabric_kwargs: Dict[str, Any] = None,
        capturing=False,
        capture_mode: str = "cum",
    ):
        """gather router

        Args:
            fabric_type (str, optional): fabric type. Defaults to "combine".
            supported keyword args for fabric:
                reduction (str, optional): reduction method. Defaults to "add".
                sparse (bool, optional): whether restore with zero paddings. Defaults to False.
                auto_padding (bool, optional): whether to pad the tensor to the max shape. Defaults to False.
        """
        super().__init__()
        self.fabric_type = fabric_type
        self.fabric_kwargs = {}

        if "combine" in self.fabric_type:
            bult_in_fabric_kwargs = {
                "flow_num": 1,
                "reduction": "add",
                "sparse": False,
                "granularity_padding": False,
            }
        self.fabric_kwargs.update(bult_in_fabric_kwargs)

        if fabric_kwargs is not None:
            self.fabric_kwargs.update(fabric_kwargs)

        self.fabric = make_fabric(fabric_type, self.fabric_kwargs)

    def forward(self, in_flows):
        out_flow = self.fabric(in_flows)
        return out_flow


@register_router("switch_gather")
class SwitchGatherRouter(RouterBase):
    def __init__(
        self,
        fabric_type: str = "combine",
        fabric_kwargs: Dict[str, Any] = None,
        capturing=False,
        capture_mode: str = "cum",
    ):
        """gather router

        Args:
            fabric_type (str, optional): fabric type. Defaults to "combine".
            supported keyword args for fabric:
                reduction (str, optional): reduction method. Defaults to "add".
                sparse (bool, optional): whether restore with zero paddings. Defaults to False.
                auto_padding (bool, optional): whether to pad the tensor to the max shape. Defaults to False.
        """
        super().__init__()
        assert fabric_type == "residual_homo_fused_combine"
        self.fabric_type = fabric_type
        self.fabric_kwargs = {}

        if "combine" in self.fabric_type:
            bult_in_fabric_kwargs = {
                "auto_pad": False,
                "flow_num": 1,
                "reduction": "add",
                "sparse": False,
                "granularity_padding": False,
            }
        self.fabric_kwargs.update(bult_in_fabric_kwargs)

        if fabric_kwargs is not None:
            self.fabric_kwargs.update(fabric_kwargs)

        self.fabric = make_fabric(fabric_type, self.fabric_kwargs)

    def forward(self, residual_flows, in_flows):
        out_flow = self.fabric(residual_flows, in_flows)
        return out_flow
