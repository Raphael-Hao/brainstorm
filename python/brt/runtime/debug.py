# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from brt.runtime.pkg_info import BRT_CACHE_PATH


class GlobalDebugerInfo:
    profile_data = []
    profile_process = "end"
    target_positions = []
    current_position = 0


def reset_global_debuger_info():
    GlobalDebugerInfo.profile_data = []
    GlobalDebugerInfo.profile_process = "end"
    GlobalDebugerInfo.target_positions = []
    GlobalDebugerInfo.current_position = 0


def set_targeted_profile_position(target: Tuple[int, List[int]], current: int = 0):
    GlobalDebugerInfo.target_positions = (
        target if isinstance(target, list) else [target]
    )
    GlobalDebugerInfo.current_position = current


def start_targeted_profile():
    GlobalDebugerInfo.profile_process = "start"


def targeted_profile(data: torch.Tensor):
    if GlobalDebugerInfo.profile_process == "end":
        return
    if GlobalDebugerInfo.current_position in GlobalDebugerInfo.target_positions:
        GlobalDebugerInfo.profile_data.append(data.cpu().numpy())
    GlobalDebugerInfo.current_position += 1


def end_targeted_profile():
    GlobalDebugerInfo.profile_process = "end"
    GlobalDebugerInfo.current_position = 0


def start_continuous_profile():
    GlobalDebugerInfo.profile_process = "start"


def continuous_profile(data: torch.Tensor):
    if GlobalDebugerInfo.profile_process == "end":
        return
    GlobalDebugerInfo.profile_data.append(data)


def end_continuous_profile():
    GlobalDebugerInfo.profile_process = "end"


def save_profile(profile_name: str = "profile", profile_dir: str = "./"):
    profile_dir_path = BRT_CACHE_PATH / profile_dir
    profile_dir_path = profile_dir_path / f"world_size_{dist.get_world_size()}"
    profile_dir_path.mkdir(parents=True, exist_ok=True)
    profile_file_path = profile_dir_path / f"rank{dist.get_rank()}_{profile_name}.npz"
    print(GlobalDebugerInfo.profile_data)
    np.savez_compressed(profile_file_path, *GlobalDebugerInfo.profile_data) # , allow_pickle=True
    reset_global_debuger_info()
