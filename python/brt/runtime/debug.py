# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import List, Tuple

import pathlib
import numpy as np
import torch
import torch.distributed as dist
from brt.runtime.pkg_info import BRT_CACHE_PATH


class GlobalDebugerInfo:
    profile_datas = []
    profile_process = "end"
    target_positions = []
    current_position = 0


def reset_global_debuger_info():
    GlobalDebugerInfo.profile_datas = []
    GlobalDebugerInfo.profile_process = "end"
    GlobalDebugerInfo.target_positions = []
    GlobalDebugerInfo.current_position = 0


def set_one_off_profile_position(target: Tuple[int, List[int]], current: int = 0):
    GlobalDebugerInfo.target_positions = (
        target if isinstance(target, list) else [target]
    )
    GlobalDebugerInfo.current_position = current


def start_one_off_profile():
    GlobalDebugerInfo.profile_process = "start"


def one_off_profile(data: torch.Tensor):
    if GlobalDebugerInfo.profile_process == "end":
        return
    if GlobalDebugerInfo.current_position in GlobalDebugerInfo.target_positions:
        GlobalDebugerInfo.profile_datas.append(data.cpu().numpy())
    GlobalDebugerInfo.current_position += 1


def end_one_off_profile():
    GlobalDebugerInfo.profile_process = "end"
    GlobalDebugerInfo.current_position = 0


def start_continuous_profile():
    GlobalDebugerInfo.profile_process = "start"


def continuous_profile(data: torch.Tensor):
    if GlobalDebugerInfo.profile_process == "end":
        return
    GlobalDebugerInfo.profile_datas.append(data)


def end_continuous_profile():
    GlobalDebugerInfo.profile_process = "end"


def save_profile(profile_name: str = "profile", profile_dir: str = "./"):
    profile_dir_path = BRT_CACHE_PATH / profile_dir
    profile_dir_path = profile_dir_path / f"world_size_{dist.get_world_size()}"
    profile_dir_path.mkdir(parents=True, exist_ok=True)
    profile_file_path = profile_dir_path / f"rank{dist.get_rank()}_{profile_name}.npy"
    np.save(profile_file_path, GlobalDebugerInfo.profile_datas)
    reset_global_debuger_info()
