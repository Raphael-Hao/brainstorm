# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

# %%
from brt.runtime.placement import generate_rank_placement

placement_file = "results_2.npy.best"

placement, placement_indices = generate_rank_placement(placement_file, 0, 2)

print(placement)
print(placement_indices)
# %%
