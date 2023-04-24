#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /sparse_a2a.py
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


from matplotlib import pyplot as plt

plt.rcParams["ytick.direction"] = "in"
plt.rcParams["hatch.linewidth"] = 2
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

# %%
import numpy as np

specu_route = np.loadtxt(RESULTS_PATH / "specu_load" / "cell_size.csv", delimiter=",")

# %%
fig, ax = plt.subplots()
fig.set_size_inches(3, 2)
x_ticks = specu_route[:, 1]
ax.plot(
    x_ticks,
    specu_route[:, 2],
    marker="*",
    mfc="white",
    color="dimgrey",
    linewidth=1.5,
    label="Default",
)
ax.plot(
    x_ticks,
    specu_route[:, 3],
    marker=".",
    mfc="white",
    color="lightseagreen",
    linewidth=1.5,
    label="Hit",
)
ax.plot(
    x_ticks,
    specu_route[:, 4],
    marker="X",
    mfc="white",
    color="tomato",
    linewidth=1.5,
    label="Miss",
)
ax.set_xticks(np.arange(0, 40, 10))
ax.set_xticklabels(np.arange(0, 40, 10), fontsize=12)
ax.set_ylim(0, 6.5)
ax.set_yticks(np.arange(0, 7, 2))
ax.set_yticklabels(np.arange(0, 7, 2), fontsize=12)
ax.set_xlabel("Weight Size (\\textit{MB})", fontsize=14, fontweight="bold")
ax.set_ylabel("Time (\\textit{ms})", fontsize=14, fontweight="bold")

ax.legend(
    ncol=3,
    loc="upper center",
    markerscale=1,
    labelspacing=0.1,
    edgecolor="black",
    shadow=False,
    fancybox=False,
    handlelength=1.2,
    handletextpad=0.6,
    columnspacing=0.8,
    prop={"weight": "bold", "size": 11},
)
plt.savefig(FIGURES_PATH / "specu_load_cell_size.pdf", dpi=300, bbox_inches="tight")

# %%
import numpy as np

specu_route = np.loadtxt(
    RESULTS_PATH / "specu_route" / "unroll_position.csv", delimiter=","
)

# %%
fig, ax = plt.subplots()
fig.set_size_inches(3, 2)
x_ticks = specu_route[:, 1]
ax.plot(
    x_ticks,
    specu_route[:, 2],
    marker="*",
    mfc="white",
    color="dimgrey",
    linewidth=1.5,
    label="Default",
)
ax.plot(
    x_ticks,
    specu_route[:, 3],
    marker=".",
    mfc="white",
    color="lightseagreen",
    linewidth=1.5,
    label="Hit",
)
ax.plot(
    x_ticks,
    specu_route[:, 4],
    marker="X",
    mfc="white",
    color="tomato",
    linewidth=1.5,
    label="Miss",
)
ax.set_xticks(np.arange(0.1, 0.5, 0.1))
ax.set_xticklabels([0.1, 0.2, 0.3, 0.4], fontsize=12)
ax.set_yticks([0.0, 0.4, 0.8])
ax.set_yticklabels([0.0, 0.4, 0.8], fontsize=12)
ax.set_ylim(0, 1.1)
ax.set_xlabel("Router Latency (\\textit{ms})", fontsize=14, fontweight="bold")
ax.set_ylabel("Time (\\textit{ms})", fontsize=14, fontweight="bold")

ax.legend(
    ncol=3,
    loc="lower center",
    markerscale=1,
    labelspacing=0.1,
    edgecolor="black",
    shadow=False,
    fancybox=False,
    handlelength=1.2,
    handletextpad=0.6,
    columnspacing=0.8,
    prop={"weight": "bold", "size": 11},
)
plt.savefig(FIGURES_PATH / "specu_route_unroll.pdf", dpi=300, bbox_inches="tight")
# %%
