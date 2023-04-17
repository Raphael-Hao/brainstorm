from typing import List, Union, Dict, Any

import torch
from torch import nn
from torch import fx

import brt
from brt.runtime import log
from brt.router import ScatterRouter, GatherRouter
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
                scatter: ScatterRouter = subm
                if (
                    # scatter.fabric.capture_mode == "max"
                    # and all(rl == "1d" for rl in scatter.fabric.route_logics)
                    scatter.fabric_type == "dispatch"
                    and scatter.fabric.load_history is not None
                ):
                    scatter.fabric.supported_capacities = torch.from_numpy(
                        scatter.fabric.load_history
                    ).to(dtype=torch.int32)
                    # scatter.fabric_type = "_fused_dispatch"
                    scatter.fabric_kwargs.update(
                        {
                            "supported_capacities": list(scatter.fabric.load_history),
                            "capacity_padding": True,
                            "path_wise_padding": True,
                        }
                    )
                    scatter.fabric.register_buffer(
                        "supported_capacities",
                        torch.tensor(
                            scatter.fabric.load_history,
                            dtype=torch.int32,
                            device="cuda",
                        ),
                    )
                    scatter.fabric.capacity_padding = True
                    scatter.fabric.path_wise_padding = True
            elif isinstance(subm, GatherRouter):
                gather: GatherRouter = subm
                if gather.fabric_type == "combine":
                    gather.fabric_kwargs.update({"ever_padded": True})
                    gather.fabric.ever_padded = True

    def finalize(self) -> GraphModule:
        return super().finalize()
