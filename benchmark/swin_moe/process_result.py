# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

#%%
import numpy as np
from itertools import product

trace_info = np.load("scatter_results.npy", allow_pickle=True)

# %%
layer_num = len(trace_info)
path_num = len(trace_info[0])
for layer_id in range(1):
    for cur_path, post_path in product(range(path_num), repeat=2):
        token_num = np.intersect1d(
            trace_info[layer_id][cur_path], trace_info[layer_id + 1][post_path]
        ).size
        print(f"layer {layer_id} path {cur_path} -> layer {layer_id + 1} path {post_path}: {token_num}")

# %%
