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

convs = dict.fromkeys([0, 2], bytes.decode)
raw_swin_moe_e2e = np.genfromtxt(
    RESULTS_PATH / "swin_moe/e2e.csv",
    dtype=[object, float, object, float],
    delimiter=",",
    converters=convs,
)

# %%

from matplotlib import pyplot as plt

# plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["hatch.linewidth"] = 2
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
# plt.rcParams["text.usetex"] = True

# %%


fig, axes = plt.subplots(3, 1)
fig.set_size_inches(6, 3.5)
plt.subplots_adjust(wspace=0, hspace=0)

total_width, n = 4, 4
group = 4
width = total_width * 0.9 / n
x = np.arange(group) * n
exit_idx_x = x + (total_width - width) / n
edgecolors = ["dimgrey", "silver", "lightseagreen", "tomato", "slategray"]
hatches = ["/////", "\\\\\\\\\\"]
labels = ["DeepSpeed", "Tutel", "BRT", "BRT+P"]
titles = ["2Gx8E", "4Gx4E", "8Gx2E"]
capacity_factors = ["1.25", "2", "3", "4"]
for i in range(3):
    row_data = raw_swin_moe_e2e[raw_swin_moe_e2e[f"f{0}"] == titles[i]]
    for j in range(n):
        bar_data = row_data[row_data[f"f{2}"] == labels[j]][f"f{3}"]
        axes[i].bar(
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
    axes[i].set_ylim([0, 480])
    axes[i].set_yticks([0, 200, 400])
    axes[i].set_yticklabels([0, 200, 400], fontsize=12)
    axes[i].set_title(titles[i], fontsize=14, y=1.0, pad=-14)
    axes[i].set_xticks([2.125, 6.125, 10.125, 14.125])
    axes[i].set_xticklabels(capacity_factors, fontsize=12)
axes[2].set_xlabel("Capacity Factors", fontsize=14)
axes[0].legend(
    bbox_to_anchor=(1, 1.4),
    ncol=4,
    # loc="upper left",
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
axes[1].set_ylabel("Throughput (Img/s)", fontsize=14)
plt.savefig(FIGURES_PATH / "figure18.pdf", bbox_inches="tight", dpi=300)

# %%
