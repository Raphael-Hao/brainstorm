from typing import List, Tuple, Union

import torch

# pylint: disable=no-name-in-module
from brt._C.router import (
    dispatch_with_dst_indices_1d,
    dispatch_with_dst_indices_2d,
    combine_with_src_indices,
)

# pylint: enable=no-name-in-module
from brt.runtime import log
from brt.router.fabric.base import register_fabric
from brt.router.fabric.generic import CombineFabric, DispatchFabric
from brt.runtime.proto_tensor import (
    ProtoTensor,
    init_proto_tensor,
    to_torch_tensor,
    make_proto_tensor_cls,
)

logger = log.get_logger(__file__)

__all__ = [
    "FusedDispatchFabric",
    "FusedCombineFabric",
]

make_proto_tensor_cls(["score"], [0])


@register_fabric("_fused_dispatch")
class FusedDispatchFabric(DispatchFabric):
    def __init__(
        self,
        flow_num: int,
        # capacity_padding=False,
        fixed_capacity: torch.Tensor,
        route_logic: Union[str, List[str]] = "1d",
        transform: Union[bool, List[bool]] = False,
    ):
        # self.capacity_padding = capacity_padding
        self.fixed_capacity = fixed_capacity
        super().__init__(
            flow_num,
            route_logic=route_logic,
            transform=transform,
            index_format="dst_index",
        )

    def dispatch(
        self,
        in_flows: List[ProtoTensor],
        route_indices: torch.Tensor,
        real_loads: torch.Tensor,
        score: torch.Tensor,
    ) -> List[List[ProtoTensor]]:
        # all_out_flows = []
        # for flow_idx, flow in enumerate(in_flows):
        #     (
        #         in_flow_data,
        #         in_flow_tag_stack,
        #         in_flow_load_stack,
        #         extra_attr_stack_dict,
        #     ) = to_torch_tensor(flow, return_stack=True)

        #     if self.route_logics[flow_idx] == "1d":
        #         if self.transforms[flow_idx]:
        #             out_flow_data = dispatch_with_dst_indices_1d(
        #                 in_flow_data, route_indices, self.fixed_capacity, False, score
        #             )
        #         else:
        #             out_flow_data = dispatch_with_dst_indices_1d(
        #                 in_flow_data, route_indices, self.fixed_capacity, False
        #             )
        #         out_flow = init_proto_tensor(
        #             out_flow_data,
        #             in_flow_tag_stack,
        #             in_flow_load_stack,
        #             extra_attr_stack_dict,
        #         )
        #         out_flow.pack(route_indices, loads, score=score)
        #     elif self.route_logics[flow_idx] == "2d":
        #         raise NotImplementedError()
        #         in_flow_data = in_flow_data.transpose(0, 1).contiguous()
        #         out_flow_data = dispatch_with_dst_indices_2d(
        #             in_flow_data, route_indices, loads, self.capacity_padding
        #         )
        #         out_flow = init_proto_tensor(
        #             out_flow_data,
        #             in_flow_tag_stack,
        #             in_flow_load_stack,
        #             extra_attr_stack_dict,
        #         )
        #         out_flow.pack(route_indices, loads)
        #     else:
        #         raise ValueError("route_logic must be 1d or 2d")
        #     all_out_flows.append(out_flow)

        # return all_out_flows
        all_out_flows = []
        path_num = route_indices.size(1)
        # real_loads = real_loads.cpu()
        route_indices_64 = route_indices.to(torch.int64)
        for flow_idx in range(self.flow_num):
            flow = in_flows[flow_idx]
            (
                flow_data,
                flow_tag_stack,
                flow_load_stack,
                extra_attr_stack_dict,
            ) = to_torch_tensor(flow, return_stack=True)

            flow_tag = flow_tag_stack[-1]
            flow_load = flow_load_stack[-1]

            flow_tag_stack = flow_tag_stack[:-1]
            flow_load_stack = flow_load_stack[:-1]
            flow_extra_attr_stack_dict = {}
            for attr_dict, attr_stack in extra_attr_stack_dict.items():
                flow_extra_attr_stack_dict[attr_dict] = attr_stack[:-1]

            if self.route_logics[flow_idx] == "1d":
                route_shape = list(flow_data.shape[1:])
                # route_size = np.prod(route_shape)
            elif self.route_logics[flow_idx] == "2d":
                # flow_data = flow_data.transpose(0, 1).contiguous()
                route_shape = list(flow_data.shape[1:])
                route_shape[0] = 1
                # route_size = np.prod(route_shape)
            else:
                raise ValueError("route_logic must be 1d or 2d")
            # out_flows = []
            # out_flow_tags = torch.gather(flow_tag, 0, route_indices_64) # TODO
            if self.route_logics[flow_idx] != "1d":
                raise NotImplementedError("currently only support 1d route_logic")
            if self.transforms[flow_idx]:
                fused_out_flow_data = dispatch_with_dst_indices_1d(
                    flow_data, route_indices, self.fixed_capacity, False, score
                )
            else:
                fused_out_flow_data = dispatch_with_dst_indices_1d(
                    flow_data, route_indices, self.fixed_capacity, False
                )
            out_flows = []
            fused_index = 0
            for i in range(path_num):
                next_index = fused_index + self.fixed_capacity[i]
                out_flow_data = fused_out_flow_data[fused_index:next_index]
                fused_index = next_index
                out_flow = init_proto_tensor(
                    out_flow_data,
                    flow_tag_stack,
                    flow_load_stack,
                    flow_extra_attr_stack_dict,
                )
                tag_indices = route_indices_64[: real_loads[i], i : i + 1]
                out_flow_tag = torch.gather(flow_tag, 0, tag_indices)
                out_flow.pack(out_flow_tag, flow_load)
                out_flows.append(out_flow)
            # for i in range(path_num):
            #     tag_indices = route_indices_64[: real_loads[i], i : i + 1]

            #     if tag_indices.numel() > 0:
            #         out_flow_tag = torch.gather(flow_tag, 0, tag_indices)
            #         # if self.route_logics[flow_idx] == "1d":
            #         #     dispatched_data = flow_data
            #         # elif self.route_logics[flow_idx] == "2d":
            #         #     raise NotImplementedError()
            #         # else:
            #         #     raise ValueError("route_logic must be 1d or 2d")
            #         if self.route_logics[flow_idx] != "1d":
            #             raise NotImplementedError("currently only support 1d route_logic")
            #         if self.transforms[flow_idx]:
            #             fused_out_flow_data = dispatch_with_dst_indices_1d(
            #                 flow_data, route_indices, self.fixed_capacity, False, score
            #             )
            #         else:
            #             fused_out_flow_data = dispatch_with_dst_indices_1d(
            #                 flow_data, route_indices, self.fixed_capacity, False
            #             )
            #         out_flow_data = fused_out_flow_data
            #         out_flow = init_proto_tensor(
            #             out_flow_data,
            #             flow_tag_stack,
            #             flow_load_stack,
            #             flow_extra_attr_stack_dict,
            #         )
            #         out_flow.pack(out_flow_tag, flow_load)
            #     else:
                    # out_flow = init_proto_tensor(
                    #     torch.zeros(
                    #         0,
                    #         *route_shape,
                    #         dtype=flow_data.dtype,
                    #         device=flow_data.device,
                    #     ),
                    #     flow_tag_stack,
                    #     flow_load_stack,
                    #     flow_extra_attr_stack_dict,
                    # )
                    # out_flow.pack(
                    #     torch.zeros(0, 1, dtype=torch.int64, device=flow_data.device),
                    #     flow_load,
                    # )
                # out_flows.append(out_flow)
            all_out_flows.append(out_flows)
        return all_out_flows

    # def pack_invalid_flow(self, in_flow):

    #     from ...runtime.proto_tensor import (
    #         ProtoTensor,
    #     )  # we need to keep ProtoTensor updated

    #     if isinstance(in_flow, ProtoTensor):
    #         pass

    #     elif isinstance(in_flow, torch.Tensor):
    #         in_flow = init_proto_tensor(in_flow)

    #     elif isinstance(in_flow, (List, Tuple)):
    #         in_flow = type(in_flow)([self.pack_invalid_flow(f) for f in in_flow])

    #     return in_flow

    # def remove_needless_pack(self, out_flow):
    #     return out_flow


@register_fabric("fused_combine")
class FusedCombineFabric(CombineFabric):
    def __init__(self, flow_num, sparse, reduction, granularity_padding) -> None:
        assert granularity_padding == False
        if flow_num != 1:
            raise NotImplementedError
        super().__init__(
            flow_num=flow_num,
            reduction=reduction,
            sparse=sparse,
            granularity_padding=False,
        )
        self.transform = True

    def combine(self, in_flows: List[ProtoTensor]) -> List[ProtoTensor]:
        out_flows = []
        for _flow_idx, in_flow in enumerate(in_flows):
            (
                in_flow_data,
                in_flow_tag_stack,
                in_flow_load_stack,
                extra_attr_stack,
            ) = to_torch_tensor(in_flow, return_stack=True)

            local_indices = in_flow_tag_stack.pop()
            loads = in_flow_load_stack.pop()
            for extra_attr_key in extra_attr_stack.keys():
                if extra_attr_key == "score_stack":
                    score = extra_attr_stack["score_stack"].pop()
                else:
                    extra_attr_stack[extra_attr_key].pop()

            # self.start_timer()
            if self.transform:
                out_flow_data = combine_with_src_indices(
                    in_flow_data, local_indices, loads, auto_pad=True, gates=score
                )
            else:
                out_flow_data = combine_with_src_indices(
                    in_flow_data, local_indices, loads, None
                )

            out_flow = init_proto_tensor(
                out_flow_data, in_flow_tag_stack, in_flow_load_stack, extra_attr_stack
            )
            out_flows.append(out_flow)

        return out_flows

    def pack_invalid_flow(self, in_flow):

        return in_flow


@register_fabric("homo_fused_dispatch")
class HomoFusedDispatchFabric(DispatchFabric):
    def __init__(
        self,
        flow_num: int,
        capacity_padding=False,
        route_logic: Union[str, List[str]] = "1d",
        transform: Union[bool, List[bool]] = False,
    ):
        self.capacity_padding = capacity_padding
        super().__init__(
            flow_num,
            route_logic=route_logic,
            transform=transform,
            index_format="dst_index",
        )

    def dispatch(
        self,
        in_flows: List[torch.Tensor],
        route_indices: torch.Tensor,
        loads: torch.Tensor,
        score: torch.Tensor,
    ) -> List[torch.Tensor]:
        all_out_flows = []
        for flow_idx, in_flow in enumerate(in_flows):
            if self.route_logics[flow_idx] == "1d":
                if self.transforms[flow_idx]:
                    out_flow = dispatch_with_dst_indices_1d(
                        in_flow, route_indices, loads, self.capacity_padding, score
                    )
                else:
                    out_flow = dispatch_with_dst_indices_1d(
                        in_flow, route_indices, loads, self.capacity_padding
                    )
                out_flow.route_indices = route_indices
                out_flow.loads = loads
                out_flow.score = score
            elif self.route_logics[flow_idx] == "2d":
                in_flow = in_flow.transpose(0, 1).contiguous()
                out_flow = dispatch_with_dst_indices_2d(
                    in_flow, route_indices, loads, self.capacity_padding
                )
                out_flow.route_indices = route_indices
                out_flow.loads = loads
            else:
                raise ValueError("route_logic must be 1d or 2d")
            all_out_flows.append(out_flow)

        return all_out_flows

    def pack_invalid_flow(self, in_flow):
        return in_flow

    def remove_needless_pack(self, out_flow):
        return out_flow


@register_fabric("homo_fused_combine")
class HomoFusedCombineFabric(CombineFabric):
    def __init__(self, flow_num, sparse, reduction, granularity_padding) -> None:
        assert granularity_padding == False
        super().__init__(
            flow_num=flow_num,
            reduction=reduction,
            sparse=sparse,
            granularity_padding=False,
        )
        self.transform = True

    def combine(self, in_flows: List[torch.Tensor]) -> List[torch.Tensor]:
        out_flows = []
        for _flow_idx, in_flow in enumerate(in_flows):

            local_indices = in_flow.route_indices
            loads = in_flow.loads
            score = in_flow.score

            # self.start_timer()
            if self.transform:
                out_flow = combine_with_src_indices(
                    in_flow, local_indices, loads, auto_pad=True, gates=score
                )
            else:
                out_flow = combine_with_src_indices(in_flow, local_indices, loads, None)


            out_flows.append(out_flow)

        return out_flows

    def pack_invalid_flow(self, in_flow):
        return in_flow

    def remove_needless_pack(self, out_flow):
        return out_flow


@register_fabric("residual_homo_fused_combine")
class ResidualHomoFusedCombineFabric(CombineFabric):
    def __init__(
        self, auto_pad, flow_num, sparse, reduction, granularity_padding
    ) -> None:
        assert granularity_padding == False
        super().__init__(
            flow_num=flow_num,
            reduction=reduction,
            sparse=sparse,
            granularity_padding=False,
        )
        self.auto_pad = auto_pad

    def forward(
        self, residual_flow: torch.Tensor, in_flow: torch.Tensor
    ) -> torch.Tensor:

        local_indices = in_flow.route_indices
        loads = in_flow.loads

        out_flow = combine_with_src_indices(
            in_flow,
            local_indices,
            loads,
            auto_pad=self.auto_pad,
            out_data=residual_flow,
        )

        return out_flow

