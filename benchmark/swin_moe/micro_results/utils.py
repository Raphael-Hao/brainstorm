# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

#%%
import numpy as np

def merge_placement(
    moe_layer_start, moe_layer_end, capacity_factor, world_size, which_one
):
    """Merge placement of different moe layers."""
    assert which_one in ["best", "worst"]
    base_file_path = f"{capacity_factor}.{which_one}_{world_size}_placement.csv"
    all_placement = []
    for i in range(moe_layer_start, moe_layer_end + 1):
        file_path = f"{i}_{i}.{base_file_path}"
        all_placement.append(np.loadtxt(file_path, delimiter=",", dtype=np.int32))
    all_placement_file = f"{moe_layer_start}_{moe_layer_end}.{base_file_path}"
    np.savetxt(all_placement_file, all_placement, delimiter=",", fmt="%d")

# %%
merge_placement(0, 8, 4.0, 8, "best")
merge_placement(0, 8, 4.0, 8, "worst")

# %%
