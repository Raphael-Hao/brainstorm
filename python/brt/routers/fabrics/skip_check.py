# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import List, Union

import numpy as np
import torch
from brt.common import log
from brt.routers.fabrics.fabric import register_fabric
from brt.routers.fabrics.generic import DispatchFabric
from brt.routers.proto_tensor import ProtoTensor, init_proto_tensor, to_torch_tensor

logger = log.get_logger(__file__)


@register_fabric("skip_check_dispatch")
class SkipCheckDispatchFabric(DispatchFabric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.default_path = kwargs.get("default_path")
        if self.default_path is None:
            self.default_path = 0
            logger.warning("default_path for skip check is not set, use 0 as default")

    def forward(
        self,
        in_flow: Union[ProtoTensor, List[ProtoTensor]],
        route_indices: torch.Tensor,
        loads: torch.Tensor,
        capacities: torch.Tensor,
        score: torch.Tensor,
    ) -> Union[List[ProtoTensor], List[List[ProtoTensor]]]:
        in_flow = self.pack_invalid_flow(in_flow)

        if isinstance(in_flow, ProtoTensor):
            in_flows = [in_flow]
        elif isinstance(in_flow, list):
            in_flows = in_flow
        else:
            raise ValueError("in_flow must be ProtoTensor or list of ProtoTensor")

        if self.throttling:
            real_loads = torch.minimum(loads, capacities)
        else:
            real_loads = loads

        all_out_flows = []

        if route_indices[-1, self.default_path] == score.size(0) - 1:
            for flow_idx, flow in enumerate(in_flows):
                (
                    flow_data,
                    flow_tag_stack,
                    flow_load_stack,
                    extra_attr_stack_dict,
                ) = to_torch_tensor(flow, copy_stack=True)

                flow_tag = flow_tag_stack.pop()
                flow_load = flow_load_stack.pop()

                for attr_stack in extra_attr_stack_dict.values():
                    attr_stack.pop()

                if self.route_logics[flow_idx] == "1d":
                    route_shape = list(flow_data.shape[1:])
                elif self.route_logics[flow_idx] == "2d":
                    flow_data = flow_data.transpose(0, 1).contiguous()
                    route_shape = list(flow_data.shape[2:])
                else:
                    raise ValueError("route_logic must be 1d or 2d")

                out_flows = []

                for i in range(self.path_num):
                    if i == self.default_path:
                        out_flow_tag = flow_tag
                        if self.route_logics[flow_idx] == "1d":
                            dispatched_data = flow_data
                        elif self.route_logics[flow_idx] == "2d":
                            dispatched_data = flow_data[i]
                        if self.transforms[flow_idx]:
                            dispatched_data = dispatched_data * score[:, i].view(
                                (-1,) + (1,) * len(route_shape)
                            )
                        out_flow_data = dispatched_data
                        out_flow = init_proto_tensor(
                            out_flow_data,
                            flow_tag_stack,
                            flow_load_stack,
                            extra_attr_stack_dict,
                        )
                        out_flow.pack(out_flow_tag, flow_load)
                    else:
                        out_flow = init_proto_tensor(
                            torch.zeros(
                                0,
                                *route_shape,
                                dtype=flow_data.dtype,
                                device=flow_data.device,
                            ),
                            flow_tag_stack,
                            flow_load_stack,
                            extra_attr_stack_dict,
                        )
                        out_flow.pack(
                            torch.zeros(
                                0, 1, dtype=torch.int64, device=flow_data.device
                            ),
                            flow_load,
                        )
                    out_flows.append(out_flow)
                all_out_flows.append(out_flows)
        else:
            all_out_flows = super().dispatch(in_flows, route_indices, real_loads, score)

        if isinstance(in_flow, ProtoTensor):
            return self.remove_needless_pack(all_out_flows[0])
        return self.remove_needless_pack(all_out_flows)
