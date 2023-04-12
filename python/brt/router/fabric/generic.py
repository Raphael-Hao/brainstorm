# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Union, Tuple

import torch
import brt._C.router as c_router
from brt.runtime import log
from brt.router.fabric.base import FabricBase, register_fabric
from brt.runtime.grid_tensor import GridTensor, init_grid_tensor, to_torch_tensor

logger = log.get_logger(__file__)

# TOD: add optimization for single cell dispatch and combine


@register_fabric("dispatch")
class DispatchFabric(FabricBase):
    def __init__(
        self,
        flow_num: int,
        route_logic: Union[str, List[str]] = "1d",
        transform: Union[bool, List[bool]] = False,
        **kwargs,
    ):
        """dispatch fabric

        Args:
            kwargs (dict):
                route_logic (str): 1d or 2d, default is 1d, can be list of 1d or 2d
                transform (bool): whether to transform input with the score, default is False, can be list of bool
        """
        index_format = kwargs.pop("index_format", "tag_index")
        super().__init__(flow_num=flow_num, index_format=index_format, **kwargs)
        route_logics = route_logic
        if isinstance(route_logics, str):
            assert route_logics in ["1d", "2d"]
            route_logics = [route_logics] * self.flow_num
        assert isinstance(route_logics, list) and all(
            isinstance(x, str) and x in ["1d", "2d"] for x in route_logics
        )
        transforms = transform
        if isinstance(transforms, bool):
            transforms = [transforms] * self.flow_num
        assert isinstance(transforms, list) and all(
            isinstance(x, bool) for x in transforms
        )

        assert len(route_logics) == len(transforms)
        assert self.flow_num == len(route_logics)
        self.route_logics = route_logics
        self.transforms = transforms
        supported_capacities = kwargs.pop("supported_capacities", None)
        if supported_capacities is not None:
            if isinstance(supported_capacities, int):
                supported_capacities = [supported_capacities]
            assert isinstance(supported_capacities, list) and all(
                isinstance(x, int) for x in supported_capacities
            )
            self.register_buffer(
                "supported_capacities",
                torch.tensor(supported_capacities, dtype=torch.int32),
            )
            self.capacity_padding = kwargs.pop("capacity_padding", False)
            self.path_wise_padding = kwargs.pop("path_wise_padding", False)
        else:
            self.register_buffer("supported_capacities", None)
            self.capacity_padding = False
            self.path_wise_padding = False
        self.max_path_padding = kwargs.pop("max_path_padding", False)
        self.max_path_load = kwargs.pop("max_path_load", None)

    def forward(
        self,
        in_flow: Union[GridTensor, List[GridTensor]],
        hot_mask: torch.Tensor,
        runtime_capacities: torch.Tensor = None,
        score: torch.Tensor = None,
    ) -> Tuple[Union[List[GridTensor], List[List[GridTensor]]], torch.Tensor]:
        supported_capacities = runtime_capacities or self.supported_capacities
        route_indices, loads = c_router.generate_indices_and_loads(
            hot_mask,
            supported_capacities=supported_capacities,
            capacity_padding=self.capacity_padding,
            path_wise_padding=self.path_wise_padding,
            is_tag_index=False,
        )
        if self.flow_num == 1:
            in_flows = [in_flow]
        else:
            in_flows = in_flow

        self.capture_flow_stats(in_flows, route_indices, loads)

        all_out_flows, score = self.dispatch(in_flows, route_indices, loads, score)
        if self.flow_num == 1:
            all_out_flows = all_out_flows[0]

        return all_out_flows, score

    def dispatch(
        self,
        in_flows: List[GridTensor],
        route_indices: torch.Tensor,
        loads: torch.Tensor,
        score: torch.Tensor,
    ) -> Tuple[List[List[GridTensor]], torch.Tensor]:
        all_out_flows = []
        path_num = route_indices.size(1)
        for flow_idx in range(self.flow_num):
            flow = in_flows[flow_idx]
            (
                flow_data,
                flow_tag_stack,
                flow_load_stack,
                extra_attr_dict,
            ) = to_torch_tensor(flow, retrieve_attr=True)

            flow_tag = flow_tag_stack[-1]
            _flow_load = flow_load_stack[-1]

            flow_tag_stack = flow_tag_stack[:-1]
            flow_load_stack = flow_load_stack[:-1]
            flow_extra_attr_dict = extra_attr_dict
            routed_results = c_router.dispatch_with_indices_and_loads(
                flow_data,
                route_indices,
                loads,
                score if self.transforms[flow_idx] else None,
                self.is_tag_index,
                flow_tag,
                self.max_path_padding,
                self.max_path_load,
                self.route_logics[flow_idx] == "1d",
            )
            if self.is_tag_index:
                routed_data, routed_tags = routed_results
                (
                    split_flow_datas,
                    split_flow_loads,
                    split_flow_tags,
                ) = c_router.split_fused_cells_to_paths(
                    routed_data,
                    loads,
                    self.max_path_padding,
                    is_load_split=True,
                    is_tag_split=True,
                    tags=routed_tags,
                )
                out_flows = []
                for i in range(path_num):
                    out_flows.append(
                        init_grid_tensor(
                            split_flow_datas[i],
                            flow_tag_stack + [split_flow_tags[i]],
                            flow_load_stack + [split_flow_loads[i]],
                            flow_extra_attr_dict,
                        )
                    )
                all_out_flows.append(out_flows)
            else:
                routed_data = routed_results
                split_flow_datas = c_router.split_fused_cells_to_paths(
                    routed_data,
                    loads,
                    self.max_path_padding,
                    is_load_split=False,
                    is_tag_split=False,
                )
                out_flows = []
                for i in range(path_num):
                    out_flows.append(
                        init_grid_tensor(
                            split_flow_datas[i],
                            flow_tag_stack + [route_indices],
                            flow_load_stack + [loads],
                            flow_extra_attr_dict,
                        )
                    )
        return all_out_flows, score


@register_fabric("combine")
class CombineFabric(FabricBase):
    def __init__(
        self,
        flow_num: int,
        reduction="add",
        transform=False,
        **kwargs,
    ):
        index_format = kwargs.pop("index_format", "tag_index")
        super().__init__(flow_num=flow_num, index_format=index_format, **kwargs)
        self.max_path_padding = kwargs.pop("max_path_padding", False)
        self.ever_padded = kwargs.pop("ever_padded", False)
        if self.max_path_padding:
            self.ever_padded = True

        self.reduction = reduction
        self.transform = transform

    def forward(
        self,
        in_flows: Union[List[GridTensor], List[List[GridTensor]]],
        residual_flows: Union[GridTensor, List[GridTensor]] = None,
        score: torch.Tensor = None,
    ) -> Union[GridTensor, List[GridTensor]]:

        if self.flow_num == 1:
            in_flows = [in_flows]
            residual_flows = [residual_flows] if residual_flows is not None else None
        self.capture_flow_stats(in_flows)
        out_flows = self.combine(in_flows, residual_flows, score)

        if self.flow_num == 1:
            return out_flows[0]

        return out_flows

    def combine(
        self,
        in_flows: List[List[GridTensor]],
        residual_flows: List[GridTensor],
        score: torch.Tensor,
    ) -> List[GridTensor]:

        out_flows = []

        for flow_idx in range(self.flow_num):
            in_flow = in_flows[flow_idx]
            residual_flow = (
                residual_flows[flow_idx] if residual_flows is not None else None
            )
            in_flows_data = []
            in_flows_tag = []
            in_flows_load = []
            route_indices = None
            for flow in in_flow:
                (
                    data,
                    flow_tags,
                    flow_loads,
                    extra_attr_dict,
                ) = to_torch_tensor(flow, retrieve_attr=True)
                in_flows_data.append(data)
                in_flows_tag.append(flow_tags[-1])
                in_flows_load.append(flow_loads[-1])
            if len(in_flows_data) == 1 and in_flows_tag[0] is None:
                in_flows_data = in_flows_data[0]
                flow_tags[-1] = torch.arange(
                    1,
                    in_flows_data.shape[0] + 1,
                    dtype=torch.int32,
                    device=in_flows_data.device,
                )
                flow_loads[-1] = torch.tensor(
                    [in_flows_data.shape[0]],
                    dtype=torch.int32,
                    device=in_flows_data.device,
                )
                out_flow = init_grid_tensor(
                    in_flows_data, flow_tags, flow_loads, extra_attr_dict
                )
                out_flows.append(out_flow)
            else:
                if self.is_tag_index:
                    (
                        in_flows_data,
                        in_flows_tag,
                        route_indices,
                    ) = c_router.fuse_split_cells_from_paths(
                        in_flows_data,
                        is_tag_fuse=True,
                        loads=in_flows_load,
                        tags=in_flows_tag,
                    )
                    in_flows_load = None
                else:
                    in_flows_data = c_router.fuse_split_cells_from_paths(in_flows_data)
                    in_flows_load = in_flows_load[0]
                    route_indices = in_flows_tag[0]
                    in_flows_tag = None
                in_flows_tag_stack = flow_tags[:-1]
                in_flows_load_stack = flow_loads[:-1]
                extra_attr_dict = extra_attr_dict
                routed_results = c_router.combine_with_indices_and_loads(
                    in_flows_data,
                    route_indices,
                    in_flows_load,
                    score if self.transform else None,
                    residual_flow,
                    max_path_padding=self.max_path_padding,
                    ever_padded=self.ever_padded,
                    is_tag_index=self.is_tag_index,
                    tags=in_flows_tag,
                )
                if self.is_tag_index:
                    out_flow_data, out_flow_tag, out_flow_load = routed_results
                else:
                    out_flow_data = routed_results
                    out_flow_tag = None
                    out_flow_load = None
                out_flow = init_grid_tensor(
                    out_flow_data,
                    in_flows_tag_stack,
                    in_flows_load_stack,
                    extra_attr_dict,
                )
                out_flow = out_flow.pack(out_flow_tag, out_flow_load)
                out_flows.append(out_flow)

        return out_flows
