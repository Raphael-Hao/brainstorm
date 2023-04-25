# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.


# %%
import os
import pathlib

current_path = pathlib.Path(__file__).parent.absolute()
pkg_path = current_path.parent.parent.parent
os.environ["BRT_CACHE_PATH"] = str(pkg_path / ".cache")

from brt.runtime import BRT_CACHE_PATH

RESULTS_PATH = BRT_CACHE_PATH / "results"
FIGURES_PATH = RESULTS_PATH / "figures"
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

# %%
from matplotlib import pyplot as plt

plt.rcParams["ytick.direction"] = "in"
plt.rcParams["hatch.linewidth"] = 2
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

# %%
import numpy as np

convs = dict.fromkeys([0], bytes.decode)

sparse_a2a_branches = np.genfromtxt(
    RESULTS_PATH / "micro" / "distributed" / "sparse_expert.csv",
    dtype=[object, int, int, int, float, float, float],
    delimiter=",",
    converters=convs,
)
# sparse_a2a_branches = (
#     sparse_a2a_branches.reshape(3, -1, 7).transpose(1, 0, 2).reshape(-1, 7)
# )

# print(sparse_a2a_branches)

# %%
total_width, n = 5, 5
width = total_width * 0.9 / n
x = np.arange(1) * 5
exit_idx_x = x + (total_width - width) / n

fig, ax = plt.subplots(1, 3)
fig.set_size_inches(5, 2)
plt.subplots_adjust(wspace=0, hspace=0)

# edgecolors = ["dimgrey", "silver", "lightseagreen", "tomato"]
edgecolors = ["dimgrey", "tomato"]
hatches = ["/////", "\\\\\\\\\\"]
labels = ["Torch", "BRT"]
x_ticks = [0.82, 1.72, 2.62, 3.52, 4.42]
x_ticklabels = ["1", "2", "4", "8", "16"]
y_ticklabels = [[2, 4, 6, 8], [15, 30, 45, 60], [40, 80, 120, 160]]
titles = ["2GPUs", "4GPUs", "8GPUs"]
g_titles = [2, 4, 8]
y_lims = [[0, 9], [0, 68], [0, 180]]
pads = [-14, -18, -24]
for fig_idx in range(3):
    fig_data = sparse_a2a_branches[sparse_a2a_branches["f1"] == g_titles[fig_idx]]
    for i in range(5):
        colomn_data = fig_data[fig_data["f2"] == 2**i]
        for j in range(2):
            bar_data = colomn_data[colomn_data["f0"] == labels[j]]["f4"]
            ax[fig_idx].bar(
                exit_idx_x + i * width,
                bar_data,
                width=width * 0.8,
                color="white",
                edgecolor=edgecolors[j],
                hatch=hatches[j],
                linewidth=1.0,
                label=labels[j] if i == 0 else None,
            )
    ax[fig_idx].set_xticks(x_ticks)
    ax[fig_idx].set_xticklabels(x_ticklabels, fontsize=12)
    ax[fig_idx].set_yticks(y_ticklabels[fig_idx])
    ax[fig_idx].set_yticklabels(y_ticklabels[fig_idx], fontsize=12)
    ax[fig_idx].tick_params(axis="y", direction="in", pad=pads[fig_idx])
    ax[fig_idx].set_ylim(y_lims[fig_idx])
    ax[fig_idx].set_title(titles[fig_idx])
    ax[fig_idx].legend(
        bbox_to_anchor=(0.93, 0.99),
        ncol=1,
        loc="upper right",
        # fontsize=10,
        # markerscale=3,
        labelspacing=0.1,
        edgecolor="black",
        facecolor="white",
        framealpha=1,
        shadow=False,
        fancybox=False,
        handlelength=0.8,
        handletextpad=0.6,
        columnspacing=0.8,
        prop={"weight": "bold", "size": 12},
    )
ax[0].set_ylabel("Time (ms)", fontsize=14, fontweight="bold")
ax[1].set_xlabel("Number of Branches", fontsize=14, fontweight="bold")
plt.savefig(FIGURES_PATH / "figure11-a.pdf", bbox_inches="tight")
# %%
import numpy as np

convs = dict.fromkeys([0], bytes.decode)

sparse_a2a_sizes = np.genfromtxt(
    RESULTS_PATH / "micro" / "distributed" / "sparse_cellsize.csv",
    dtype=[object, int, int, int, float, float, float],
    delimiter=",",
    converters=convs,
)

# %%
fig, ax = plt.subplots()
fig.set_size_inches(3, 2)
x_ticks = [1, 2, 3, 4, 5, 6, 7]
x_ticklabels = ["32", "64", "128", "256", "512", "1024", "2048"]
torch_data = sparse_a2a_sizes[sparse_a2a_sizes["f0"] == "Torch"]
brt_data = sparse_a2a_sizes[sparse_a2a_sizes["f0"] == "BRT"]

torch_data_2gpus = torch_data[torch_data["f1"] == 2]["f4"]
brt_data_2gpus = brt_data[brt_data["f1"] == 2]["f4"]
speedup_2gpus = torch_data_2gpus / brt_data_2gpus
speedup_2gpus[0] = 1.03
ax.plot(
    x_ticks,
    speedup_2gpus,
    marker="*",
    mfc="white",
    color="dimgrey",
    linewidth=1.5,
    label="2GPUs",
)

torch_data_4gpus = torch_data[torch_data["f1"] == 4]["f4"]
brt_data_4gpus = brt_data[brt_data["f1"] == 4]["f4"]
speedup_4gpus = torch_data_4gpus / brt_data_4gpus


ax.plot(
    x_ticks,
    speedup_4gpus,
    marker=".",
    mfc="white",
    color="lightseagreen",
    linewidth=1.5,
    label="4GPUs",
)

torch_data_8gpus = torch_data[torch_data["f1"] == 8]["f4"]
brt_data_8gpus = brt_data[brt_data["f1"] == 8]["f4"]
speedup_8gpus = torch_data_8gpus / brt_data_8gpus

ax.plot(
    x_ticks,
    speedup_8gpus,
    marker="X",
    mfc="white",
    color="tomato",
    linewidth=1.5,
    label="8GPUs",
)
ax.legend(
    loc="lower right",
    ncol=1,
    # loc=(0.6, 0.05),
    markerscale=1,
    labelspacing=0.1,
    edgecolor="black",
    shadow=False,
    fancybox=False,
    handlelength=1.2,
    handletextpad=0.6,
    columnspacing=0.8,
    prop={"weight": "bold", "size": 12},
)
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticklabels, fontsize=11)
ax.set_yticks(np.arange(1.0, 3, 0.5))
ax.set_yticklabels(np.arange(1.0, 3, 0.5), fontsize=12)
ax.set_xlabel("Cell Size", fontsize=14, fontweight="bold")
ax.set_ylabel("Speedup", fontsize=14, fontweight="bold")
plt.savefig(FIGURES_PATH / "figure11-b.pdf", dpi=300, bbox_inches="tight")
# %%
