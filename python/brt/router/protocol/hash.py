# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.distributed as dist
from brt.runtime import log
from brt.router.utils import generate_dst_indices
from brt.router.protocol.base import ProtocolBase, register_protocol

__all__ = ["HashProtocol"]

logger = log.get_logger(__file__)


@register_protocol("hash")
class HashProtocol(ProtocolBase):
    def __init__(
        self,
        num_tasks: int,
        placement_aware: bool = False,
        seed: int = 0,
        supported_capacities: torch.Tensor = None,
        index_format="dst_index",
        index_gen_opt=True,
    ):
        """Top-K protocol

        Args:
            top_k (int, optional): k for top selecting. Defaults to 1.
            supported_capacities (optional): _description_. Defaults to None.
            index_format (str, optional): index tensors according to destination or source. Defaults to "src_index".
            index_gen_opt (bool, optional): whether use optimized GPU kernel. Defaults to True.
        """
        super().__init__(index_format=index_format, index_gen_opt=index_gen_opt)
        self.supported_capacities = supported_capacities
        self.num_tasks = num_tasks
        self.placement_aware = placement_aware
        if self.placement_aware:
            hash_indices_1 = torch.arange(num_tasks, dtype=torch.int64).view(-1, 1)
            hash_indices_2 = (
                torch.arange(num_tasks, dtype=torch.int64).view(-1, 1) + 1
            ) % num_tasks
        else:
            torch.random.manual_seed(seed)
            hash_indices_1 = torch.randperm(num_tasks, dtype=torch.int64).view(-1, 1)
            hash_indices_2 = torch.randperm(num_tasks, dtype=torch.int64).view(-1, 1)
        hash_indices = torch.cat([hash_indices_1, hash_indices_2], dim=1)
        self.register_buffer("hash_indices", hash_indices)
        self.check_hash_indices()

    def make_route_decision(self, task_ids: torch.Tensor):
        hash_dest = self.hash_indices[task_ids]
        hot_mask = torch.zeros_like(
            (task_ids.size(0), self.num_tasks),
            dtype=torch.int64,
            device=task_ids.device,
        ).scatter_(1, hash_dest, 1)
        route_indices, loads = generate_dst_indices(
            hot_mask, self.supported_capacities, self.index_format, self.index_gen_opt
        )
        capacity = loads.max()
        capacity = dist.all_reduce(capacity, op=dist.ReduceOp.MAX)
        route_indices.capacity = capacity
        return route_indices, loads, loads

    def check_hash_indices(self):
        assert self.hash_indices.size(0) == self.num_tasks
        assert self.hash_indices.size(1) == 2


@register_protocol("task")
class TaskProtocol(ProtocolBase):
    def __init__(
        self,
        num_tasks: int,
        supported_capacities: torch.Tensor = None,
        index_format="dst_index",
        index_gen_opt=True,
    ):
        """Top-K protocol

        Args:
            top_k (int, optional): k for top selecting. Defaults to 1.
            supported_capacities (optional): _description_. Defaults to None.
            index_format (str, optional): index tensors according to destination or source. Defaults to "src_index".
            index_gen_opt (bool, optional): whether use optimized GPU kernel. Defaults to True.
        """
        super().__init__(index_format=index_format, index_gen_opt=index_gen_opt)
        self.supported_capacities = supported_capacities
        self.num_tasks = num_tasks
        task_indices = torch.arange(num_tasks, dtype=torch.int64).view(-1, 1)
        self.register_buffer("task_indices", task_indices)

    def make_route_decision(self, task_ids: torch.Tensor):
        task_dest = self.task_indices[task_ids]
        hot_mask = torch.zeros_like(
            (task_ids.size(0), self.num_tasks),
            dtype=torch.int64,
            device=task_ids.device,
        ).scatter_(1, task_dest, 1)
        route_indices, loads = generate_dst_indices(
            hot_mask, self.supported_capacities, self.index_format, self.index_gen_opt
        )
        capacity = loads.max()
        capacity = dist.all_reduce(capacity, op=dist.ReduceOp.MAX)
        route_indices.capacity = capacity
        return route_indices, loads, loads
