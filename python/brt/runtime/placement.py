# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import pathlib
from typing import Dict, List, Tuple

from collections import OrderedDict
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from brt.router import is_router
from brt.runtime import log

logger = log.get_logger(__file__)


"""
Currently all placement-related optimizations are implemented for Swin-MoE.
In the future, we have to generalize the placement-related optimizations to
all dynamic models.
"""


def dump_trace(mod: nn.Module):
    scatter_results = []
    for _m_name, m in mod.named_modules():
        if is_router(m) and "scatter" in m._router_type:
            scatter_results.append(np.array(m.ptu_decision_history, dtype=object))
    np.save("scatter_results.npy", scatter_results, allow_pickle=True)


def generate_experts_keys(experts_range: Dict[int, int]):
    experts_keys: List[Tuple[int, int]] = []
    for layer_id, block_num in experts_range.items():
        for block_id in range(1, block_num, 2):
            experts_keys.append((layer_id, block_id))
    return experts_keys


def generate_rank_placement(
    world_rank: int,
    world_size: int,
    placement_file: str = None,
    global_expert_num: int = None,
):
    experts_range = {2: 18, 3: 2}
    experts_keys = generate_experts_keys(experts_range)

    assert placement_file is not None or global_expert_num is not None
    all_placement: List[np.ndarray] = []
    if placement_file is None:
        assert global_expert_num % world_size == 0
        local_expert_num = global_expert_num // world_size
        placement = np.repeat(np.arange(world_size), local_expert_num)
        all_placement = [placement for i in range(len(experts_keys))]
    else:
        all_placement = np.load(placement_file, allow_pickle=True)
    rank_placement: OrderedDict[Tuple[int, int], List[int]] = OrderedDict()
    placement_indices: OrderedDict[Tuple[int, int], torch.Tensor] = OrderedDict()
    for idx, placement in enumerate(all_placement):
        expert_key = experts_keys[idx]
        placement_index = []
        for rank_idx in range(world_size):
            placement_index.append(np.where(placement == rank_idx)[0])
            if rank_idx == world_rank:
                rank_placement[expert_key] = placement_index[-1].tolist()
        placement_index = np.concatenate(placement_index, axis=None)
        placement_indices[expert_key] = torch.from_numpy(placement_index)

    # if placement_file is None:
    #     placement_indices = None

    return rank_placement, placement_indices


def adaptive_load(
    model: nn.Module,
    checkpoint_file: str,
    enable_locality=False,
    placement_file: str = None,
    global_expert_num: int = None,
):
    world_rank = dist.get_rank()
    world_size = dist.get_world_size()
    rank_placement, placement_indices = generate_rank_placement(
        world_rank, world_size, placement_file, global_expert_num
    )
    adaptive_load_checkpoint(model, checkpoint_file, rank_placement)
    # adaptive_load_placement(model, placement_indices)
    print(f"rank {world_rank} placement_indices is {placement_indices}")
    locality_enabled_router = []
    if enable_locality:
        locality_enabled_router = [
            next(iter(placement_indices.keys())),
            next(reversed(placement_indices.keys())),
        ]
        print(f"locality_enabled_router: {locality_enabled_router}")
    set_locality(model, locality_enabled_router)


def adaptive_load_placement(
    model: nn.Module,
    placement_indices: "OrderedDict[Tuple[int, int], torch.Tensor]" = None,
):
    # layers.{layer_id}.blocks.{block_id}.mlp._moe_layer.scatter
    if placement_indices is None:
        return
    for _m_name, m in model.named_modules():
        if is_router(m) and "scatter" in m._router_type:
            expert_key = tuple([int(_m_name.split(".")[i]) for i in [1, 3]])
            print(
                f"setting placement for {_m_name} with {placement_indices[expert_key]}"
            )
            m.fabric.placement_indices = placement_indices[expert_key]


def adaptive_load_checkpoint(
    model: nn.Module,
    checkpoint_file: str,
    rank_placement: "OrderedDict[Tuple[int, int], List[int]]",
):
    ckpt_filepath = pathlib.Path(checkpoint_file)
    checkpoint = torch.load(ckpt_filepath, map_location="cpu")
    all_expert_state_dict = {}
    state_dict = {}
    for k in checkpoint.keys():
        if "._moe_layer.experts." in k:
            k_var_id = k.split("._moe_layer.experts.0.")
            k_list = k_var_id[0].split(".")
            var_name, expert_id = k_var_id[1].split(".")
            expert_k = (int(k_list[1]), int(k_list[3]))
            expert_id = int(expert_id)
            if expert_k not in all_expert_state_dict:
                all_expert_state_dict[expert_k] = {}
            if expert_id not in all_expert_state_dict[expert_k]:
                all_expert_state_dict[expert_k][expert_id] = {}
            all_expert_state_dict[expert_k][expert_id][var_name] = checkpoint[k]
        else:
            state_dict[k] = checkpoint[k]

    for k in rank_placement.keys():
        tensors_to_concat = {}
        for expert_id in rank_placement[k]:
            tensor_entries = all_expert_state_dict[k][expert_id].keys()
            for entry in tensor_entries:
                full_entry_name = (
                    f"layers.{k[0]}.blocks.{k[1]}.mlp._moe_layer.experts.0.{entry}"
                )
                if full_entry_name not in tensors_to_concat:
                    tensors_to_concat[full_entry_name] = []
                tensors_to_concat[full_entry_name].append(
                    all_expert_state_dict[k][expert_id][entry]
                )
        for entry in tensors_to_concat.keys():
            if len(tensors_to_concat[entry]) == 1:
                state_dict[entry] = tensors_to_concat[entry][0]
            else:
                if "_bias" in entry:
                    state_dict[entry] = torch.stack(
                        [torch.unsqueeze(x, dim=0) for x in tensors_to_concat[entry]]
                    )
                else:
                    state_dict[entry] = torch.stack(tensors_to_concat[entry])

    msg = model.load_state_dict(state_dict, strict=False)
    # logger.info(msg)
    del checkpoint
    torch.cuda.empty_cache()


def set_locality(model: nn.Module, locality_enabled_router: List[Tuple[int, int]]):
    for _m_name, m in model.named_modules():
        if is_router(m) and "scatter" in m._router_type:
            expert_key = tuple([int(_m_name.split(".")[i]) for i in [1, 3]])
            if expert_key in locality_enabled_router:
                print(
                    f"orginal locality for {_m_name}.fabric is {m.fabric.locality_aware}"
                )
                m.fabric.locality_aware = True
