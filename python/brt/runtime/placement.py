# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Tuple
import torch.nn as nn
from brt.router import is_router
import numpy as np


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


def generate_rank_placement(placement_file, world_rank: int, world_size: int):
    all_placement: Dict[str, np.ndarray] = np.load(placement_file, allow_pickle=True)
    experts_range = {2: 18, 3: 2}
    experts_keys = generate_experts_keys(experts_range)

    rank_placement = {}
    placement_indices = {}
    for m_name, placement in all_placement.items():
        placement_index = []
        for rank_idx in range(world_size):
            placement_index.append(np.where(placement == rank_idx))
            if rank_idx == world_rank:
                rank_placement[m_name] = placement_index[-1]
        placement_index = np.concatenate(placement_index, axis=None)
        placement_indices[m_name] = placement_index
    return rank_placement, placement_indices
