# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

#%%
from itertools import product
import numpy as np

trace_info = np.load("scatter_results.npy", allow_pickle=True)

layer_num = len(trace_info)
path_num = len(trace_info[0])
scatter_trace = []

for layer_id in range(layer_num - 1):
    scatter_trace.append(np.empty((path_num, path_num), dtype=np.int32))
    for cur_path, post_path in product(range(path_num), repeat=2):
        token_num = np.intersect1d(
            trace_info[layer_id][cur_path], trace_info[layer_id + 1][post_path]
        ).size
        # print(f"layer {layer_id} path {cur_path} -> layer {layer_id + 1} path {post_path} token num {token_num}")
        scatter_trace[layer_id][cur_path, post_path] = token_num

np.save("scatter_trace.npy", scatter_trace, allow_pickle=True)

# %%
scatter_trace = np.load("scatter_trace.npy", allow_pickle=True)
import seaborn as sns
from matplotlib import pyplot as plt

for i in range(len(scatter_trace)):
    plt.figure()
    ax = sns.heatmap(scatter_trace[i], linewidth=0.5, cmap="YlGnBu")
    plt.savefig(f"figures/scatter_trace_{i}.png")


# %%
