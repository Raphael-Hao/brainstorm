# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

# %%
from brt.runtime.placement import generate_rank_placement

placement_file = "/home/zhehan/brainstorm_project/brainstorm/benchmark/swin_moe/placement.2.best"

placement, placement_indices = generate_rank_placement(0, 2, placement_file)
print(placement)
print(placement_indices)
placement, placement_indices = generate_rank_placement(0, 2, placement_file)
print(placement)
print(placement_indices)

# %%
