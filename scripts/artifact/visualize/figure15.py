#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /switch_transformer.py
# \brief:
# Author: raphael hao

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

convs = dict.fromkeys([0], bytes.decode)
switch_e2e = np.genfromtxt(
    RESULTS_PATH / "switch_moe/e2e.csv",
    dtype=[object, int, float],
    delimiter=",",
    converters=convs,
)
# print(np.append(switch_e2e[switch_e2e[f"f{0}"] == "Tutel"][f"f{2}"], 0))

# %%

from matplotlib import pyplot as plt

plt.rcParams["ytick.direction"] = "in"
plt.rcParams["hatch.linewidth"] = 2
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
# %%

fig, ax = plt.subplots()
fig.set_size_inches(5, 1.5)
plt.subplots_adjust(wspace=0, hspace=0)

total_width, n = 3, 3
group = 6
width = total_width * 0.9 / n
x = np.arange(group) * n
exit_idx_x = x + (total_width - width) / n
edgecolors = ["dimgrey", "lightseagreen", "tomato", "silver", "slategray"]
hatches = ["/////", "\\\\\\\\\\"]
labels = ["Torch", "Tutel", "BRT"]
for j in range(n):
    bar_data = switch_e2e[switch_e2e[f"f{0}"] == labels[j]]["f2"]
    if labels[j] == "Tutel":
        bar_data = np.append(bar_data, 0)
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
# ax.set_ylim([0, 95])
ax.set_yticks(np.arange(0, 700, 200))
ax.set_yticklabels(np.arange(0, 700, 200), fontsize=12)
ax.set_xticks(exit_idx_x + width)
ax.set_xticklabels([8, 16, 32, 64, 128, 256], fontsize=12)
ax.set_xlabel("Number of Experts", fontsize=14)
ax.text(x=16.43, y=16, s="x", fontsize=10, fontweight="bold", zorder=100)
ax.legend(
    # bbox_to_anchor=(1.02, 1.3),
    ncol=3,
    loc="upper left",
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
ax.set_ylabel("Time (ms)", fontsize=14)
plt.savefig(FIGURES_PATH / "figure15.pdf", bbox_inches="tight", dpi=300)

# %%
