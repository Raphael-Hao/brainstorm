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

swin_moe_8g2e_caps = np.loadtxt(
    RESULTS_PATH / "swin_moe/micro_throughput.csv", delimiter=","
)


# %%
from matplotlib import pyplot as plt

plt.rcParams["ytick.direction"] = "in"
plt.rcParams["hatch.linewidth"] = 2
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

# %%

fig, ax = plt.subplots()
fig.set_size_inches(5, 2)
plt.subplots_adjust(wspace=0, hspace=0)

# edgecolors = ["dimgrey", "silver", "lightseagreen", "tomato"]
colors = ["dimgrey", "silver", "lightseagreen", "tomato"]
hatches = ["/////", "\\\\\\\\\\"]
markers = ["*", ".", "X", "P"]
x_ticks = np.arange(1, 11)
# x_ticklabels = ["32", "64", "96", "128", "160", "192", "224", "256"]
labels = [1.25, 2, 3, 4]

for i in range(4):
    plot_data = swin_moe_8g2e_caps[swin_moe_8g2e_caps[:, 0] == labels[i]][:, 2]
    print(plot_data)
    ax.plot(
        x_ticks,
        plot_data,
        marker=markers[i],
        mfc="white",
        color=colors[i],
        linewidth=1.5,
        label=labels[i],
    )

# ax_1 = ax.twinx()
# ax_1.plot(
#     x_ticks,
#     swin_moe_8g2e_caps[:, 4],
#     marker="X",
#     mfc="white",
#     color="lightseagreen",
#     linewidth=1.5,
#     label="Speedup",
# )
# ax_1.set_ylabel("Speedup", fontsize=14, fontweight="bold")
# ax_1.set_yticks([0.0, 1.0, 2.0],fontsize=12)
# ax_1.set_yticks([1.5, 2.0, 2.5])
# ax.set_yticks(y_ticklabels[fig_idx], fontsize=12)
# ax.tick_params(axis="y", direction="in", pad=pads[fig_idx])
# ax.set_ylim(y_lims[fig_idx])
# ax.set_title()
ax.legend(
    # bbox_to_anchor=(0.16, 0.9),
    ncol=4,
    loc="upper left",
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
ax.set_ylim(1, 1.6)
ax.set_ylabel("Speedup", fontsize=14, fontweight="bold")
ax.set_xlabel("Layer ID", fontsize=14, fontweight="bold")
ax.set_xticks(x_ticks)
ax.set_xticklabels(range(1, 11), fontsize=12)
ax.set_yticks([1.0, 1.2, 1.4, 1.6])
ax.set_yticklabels([1.0, 1.2, 1.4, 1.6], fontsize=12)
plt.savefig(FIGURES_PATH / "figure19.pdf", bbox_inches="tight")

# %%
