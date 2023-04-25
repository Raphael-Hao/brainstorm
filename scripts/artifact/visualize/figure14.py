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

convs = dict.fromkeys([0], bytes.decode)
specu_route = np.genfromtxt(
    RESULTS_PATH / "micro/speculative/load_e2e.csv",
    dtype=[object, float, float, float, float],
    delimiter=",",
    converters=convs,
)

# %%
fig, ax = plt.subplots()
fig.set_size_inches(3, 2)
plot_data = specu_route[specu_route["f0"] == "Default"]
x_ticks = plot_data["f1"]
y_data = plot_data["f2"]
ax.plot(
    x_ticks,
    y_data,
    marker="*",
    mfc="white",
    color="dimgrey",
    linewidth=1.5,
    label="Default",
)
plot_data = specu_route[specu_route["f0"] == "Hit"]
y_data = plot_data["f2"]
ax.plot(
    x_ticks,
    y_data,
    marker=".",
    mfc="white",
    color="lightseagreen",
    linewidth=1.5,
    label="Hit",
)
plot_data = specu_route[specu_route["f0"] == "Miss"]
y_data = plot_data["f2"]
ax.plot(
    x_ticks,
    y_data,
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
ax.set_xlabel("Weight Size (MB)", fontsize=14, fontweight="bold")
ax.set_ylabel("Time (ms)", fontsize=14, fontweight="bold")

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
plt.savefig(FIGURES_PATH / "figure14-b.pdf", dpi=300, bbox_inches="tight")

# %%
import numpy as np

convs = dict.fromkeys([0], bytes.decode)
specu_route = np.genfromtxt(
    RESULTS_PATH / "micro/speculative/route_e2e.csv",
    dtype=[object, float, float, float, float],
    delimiter=",",
    converters=convs,
)

# %%
fig, ax = plt.subplots()
fig.set_size_inches(3, 2)
plot_data = specu_route[specu_route["f0"] == "Default"]
x_ticks = plot_data["f1"]
y_data = plot_data["f2"]
ax.plot(
    x_ticks,
    y_data,
    marker="*",
    mfc="white",
    color="dimgrey",
    linewidth=1.5,
    label="Default",
)
plot_data = specu_route[specu_route["f0"] == "Hit"]
y_data = plot_data["f2"]
ax.plot(
    x_ticks,
    y_data,
    marker=".",
    mfc="white",
    color="lightseagreen",
    linewidth=1.5,
    label="Hit",
)
plot_data = specu_route[specu_route["f0"] == "Miss"]
y_data = plot_data["f2"]
ax.plot(
    x_ticks,
    y_data,
    marker="X",
    mfc="white",
    color="tomato",
    linewidth=1.5,
    label="Miss",
)
ax.set_xticks(np.arange(0.1, 0.5, 0.1))
ax.set_xticklabels([0.1, 0.2, 0.3, 0.4], fontsize=12)
#ax.set_yticks([0.0, 0.4, 0.8])
#ax.set_yticklabels([0.0, 0.4, 0.8], fontsize=12)
ax.set_ylim(0, 1.5)
ax.set_xlabel("Router Latency (ms)", fontsize=14, fontweight="bold")
ax.set_ylabel("Time (ms)", fontsize=14, fontweight="bold")

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
plt.savefig(FIGURES_PATH / "figure14-a.pdf", dpi=300, bbox_inches="tight")
# %%
