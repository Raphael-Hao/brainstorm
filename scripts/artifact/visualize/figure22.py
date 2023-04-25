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

convs = dict.fromkeys([0, 1], bytes.decode)
dr_load_memory = np.genfromtxt(
    RESULTS_PATH / "dynamic_routing/speculative_load_memory.csv",
    dtype=[object, object, float],
    delimiter=",",
    converters=convs,
)

# %%

fig, ax = plt.subplots()
fig.set_size_inches(3, 2)
plt.subplots_adjust(wspace=0, hspace=0)

total_width, n = 2, 2
group = 4
width = total_width * 0.9 / n
x = np.arange(group) * n
exit_idx_x = x + (total_width - width) / n
edgecolors = ["dimgrey", "tomato", "silver", "lightseagreen", "slategray"]
hatches = ["/////", "\\\\\\\\\\"]
labels = [
    "On-demand",
    "Speculative",
    "Original",
]
for j in range(n):
    bar_data = dr_load_memory[dr_load_memory["f0"] == labels[j]]["f2"]
    ax.bar(
        exit_idx_x + j * width,
        bar_data,
        width=width * 0.8,
        color="white",
        edgecolor=edgecolors[j],
        hatch=hatches[j % 2],
        linewidth=1.0,
        label=labels[j],
        zorder=10,
    )
ax.set_ylim([0, 700])
original_memory = dr_load_memory[dr_load_memory["f0"] == labels[2]]["f2"][0]
ax.axhline(
    y=original_memory,
    color="red",
    linestyle="--",
    linewidth=1.0,
)
ax.text(
    x=0.2,
    y=original_memory + 20,
    s="Original",
    color="red",
    fontsize=12,
    zorder=100,
)
ax.set_yticks(np.arange(0, 700, 200), fontsize=12)
ax.set_yticklabels(np.arange(0, 700, 200), fontsize=12)
ax.set_ylim(0, 700)
# # ax.set_title(dr_load_e2e["f0"], fontsize=14)
ax.set_xticks([1, 3, 5, 7])
ax.set_xticklabels(["A", "B", "C", "Raw"], fontsize=12)
ax.set_xlabel("Model Architectures", fontsize=14)
ax.legend(
    # bbox_to_anchor=(1.02, 1.3),
    ncol=1,
    loc="lower right",
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
).set_zorder(100)
ax.set_ylabel("Memory (MB)", fontsize=14)
plt.savefig(FIGURES_PATH / "figure22-b.pdf", bbox_inches="tight", dpi=300)

# %%


# %%

# %%
import numpy as np

convs = dict.fromkeys([0], bytes.decode)
dr_load_latency = np.genfromtxt(
    RESULTS_PATH / "dynamic_routing/speculative_load_time.csv",
    dtype=[object, object, float, float, float],
    delimiter=",",
    converters=convs,
)

# %%

fig, ax = plt.subplots()
fig.set_size_inches(3, 2)
plt.subplots_adjust(wspace=0, hspace=0)

total_width, n = 2, 2
group = 4
width = total_width * 0.9 / n
x = np.arange(group) * n
exit_idx_x = x + (total_width - width) / n
edgecolors = ["dimgrey", "tomato", "silver", "lightseagreen", "slategray"]
hatches = ["/////", "\\\\\\\\\\"]
labels = [
    "On-demand",
    "Speculative",
]
for j in range(n):
    bar_data = dr_load_latency[dr_load_latency["f0"] == labels[j]]["f2"]
    ax.bar(
        exit_idx_x + j * width,
        bar_data,
        width=width * 0.8,
        color="white",
        edgecolor=edgecolors[j],
        hatch=hatches[j % 2],
        linewidth=1.0,
        label=labels[j],
        zorder=10,
    )
# ax.set_ylim([0, 700])
# ax.axhline(y=dr_load_latency[0]["f0"], color="red", linestyle="--", linewidth=1.0)
# ax.set_yticks(np.arange(0, 700, 200), fontsize=12)
# ax.set_yticklabels(np.arange(0, 700, 200), fontsize=12)
# # ax.set_title(dr_load_e2e["f0"], fontsize=14)
ax.set_xticks([1, 3, 5, 7])
ax.set_xticklabels(["A", "B", "C", "Raw"], fontsize=12)
ax.set_yticks(np.arange(0, 130, 30))
ax.set_yticklabels(np.arange(0, 130, 30), fontsize=12)
ax.set_xlabel("Model Architectures", fontsize=14)
ax.legend(
    # bbox_to_anchor=(1.02, 1.3),
    ncol=1,
    loc="lower right",
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
).set_zorder(100)
ax.set_ylabel("Latency (ms)", fontsize=14)
plt.savefig(FIGURES_PATH / "figure22-a.pdf", bbox_inches="tight", dpi=300)
