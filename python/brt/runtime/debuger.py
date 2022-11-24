# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch.distributed as dist

class GlobalDebuger():
    profile_datas = []
    profile_process = "end"

def one_off_profile(data):
    if GlobalDebuger.profile_process == "end":
        GlobalDebuger.profile_datas.append(data)
        GlobalDebuger.profile_process = "start"

def continuous_profile(data):
    GlobalDebuger.profile_datas.append(data)
    if GlobalDebuger.profile_process == "end":
        GlobalDebuger.profile_process = "start"

def end_profile():
    GlobalDebuger.profile_process = "end"

def save_profile():
    np.save(f"rank{dist.get_rank()}_profile.npy", GlobalDebuger.profile_datas)
    GlobalDebuger.profile_process = "end"