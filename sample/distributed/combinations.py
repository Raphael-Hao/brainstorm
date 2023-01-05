# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

#%%
from brt.runtime.placement import adaptive_micro_bench_load, posible_placement_generator

possible_placement = posible_placement_generator(16, 2)
print(possible_placement)
print(len(possible_placement))

# %%
adaptive_micro_bench_load(
    None, possible_placement[2], (2, 1), checkpoint_file=None, global_expert_num=16
)

# %%
