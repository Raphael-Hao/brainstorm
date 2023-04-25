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
import numpy as np

convs = dict.fromkeys([0, 1, 2], bytes.decode)
task_moe_e2e = np.genfromtxt(
    RESULTS_PATH / "task_moe/throughput.csv",
    dtype=[object, object, object, float],
    delimiter=",",
    converters=convs,
)

total_width, n = 3, 3
width = total_width * 0.9 / n
x = np.arange(3) * n
exit_idx_x = x + (total_width - width) / n

# %%
from matplotlib import pyplot as plt

plt.rcParams["ytick.direction"] = "in"
plt.rcParams["hatch.linewidth"] = 2
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


# %%
fig, axs = plt.subplots(1, 3)
fig.set_size_inches(8, 1.5)
plt.subplots_adjust(wspace=0, hspace=0)
tiles = ["256Seqsx32Tokens", "256Seqsx64Tokens", "512Seqsx32Tokens"]
for i in range(3):
    all_bar_data = task_moe_e2e[task_moe_e2e["f2"] == tiles[i]]
    bar_data = all_bar_data[all_bar_data["f0"] == "Torch"]["f3"]
    axs[i].bar(
        exit_idx_x,
        bar_data,
        width=width * 0.8,
        color="white",
        edgecolor="dimgrey",
        hatch="/////",
        linewidth=1.0,
        label="Torch",
        zorder=10,
    )
    bar_data = all_bar_data[all_bar_data["f0"] == "BRT"]["f3"]
    axs[i].bar(
        exit_idx_x + width,
        bar_data,
        width=width * 0.8,
        color="white",
        edgecolor="lightseagreen",
        hatch="/////",
        linewidth=1.0,
        label="BRT",
        zorder=10,
    )
    bar_data = all_bar_data[all_bar_data["f0"] == "BRT+P"]["f3"]
    axs[i].bar(
        exit_idx_x + 2 * width,
        bar_data,
        width=width * 0.8,
        color="white",
        edgecolor="tomato",
        hatch="\\\\\\\\\\",
        linewidth=1.0,
        label="BRT+P",
        zorder=10,
    )
    axs[i].set_ylim(0, 2550)
    if i == 0:
        axs[i].set_ylabel("Throughput (Seq/s)", fontsize=14)
        axs[i].set_yticks([0, 1000, 2000])
        axs[i].set_yticklabels(["0", "1k", "2k"], fontsize=12)
    else:
        axs[i].set_yticks([0, 1000, 2000])
        axs[i].set_yticklabels([])
    axs[i].set_xticks(exit_idx_x + width)
    axs[i].set_xticklabels(["2GPUx8E", "4GPUx4E", "8GPUx2E"], rotation=20, fontsize=12)
    axs[i].set_title(tiles[i], fontsize=14)

axs[2].legend(
    # bbox_to_anchor=(0.08, 1),
    ncol=3,
    loc="lower right",
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
).set_zorder(100)

plt.savefig(FIGURES_PATH / "figure17.pdf", bbox_inches="tight", dpi=300)
