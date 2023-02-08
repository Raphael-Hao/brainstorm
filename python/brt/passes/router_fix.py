from typing import List, Union, Dict, Any

import torch
from torch import nn
from torch import fx

import brt
from brt.runtime import log
from brt.router import ScatterRouter
from brt.router.fabric import make_fabric
from brt.router.protocol import make_protocol
from brt.trace.graph import symbolic_trace, GraphModule, Graph, Node

from brt.passes.base import PassBase, register_pass
from brt.passes.utils import build_sub_graph

logger = log.get_logger(__file__)


@register_pass("router_fix")
class RouterFixPass(PassBase):
    def __init__(self, m: Union[torch.nn.Module, GraphModule]):
        super().__init__(m)
    
    def run_on_graph(self) -> None:
        # for node in self.graph_mod.graph.nodes:
        for subm in self.sub_modules.values():
            if isinstance(subm, ScatterRouter):
                router: ScatterRouter = subm
                if (
                    router.capturing is True
                    and "dispatch" in router.captured_fabric_type
                    and router.capture_mode == "max"
                    and all(rl == "1d" for rl in router.fabric.route_logics)
                ):
                    if (
                        router.fabric_type == "dispatch"
                        and router.load_history is not None
                    ):
                        router.fabric_type = "_fused_dispatch"
                        router.fabric_kwargs.update(
                            {
                                "fixed_capacity": torch.from_numpy(
                                    router.load_history
                                )
                                .to(torch.int32)
                                .cuda()
                            }
                        )
                        router.fabric = make_fabric(
                            "_fused_dispatch", router.fabric_kwargs
                        )


    def finalize(self) -> GraphModule:
        return super().finalize()