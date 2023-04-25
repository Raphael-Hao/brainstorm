import os
import pathlib

current_path = pathlib.Path(__file__).parent.absolute()
pkg_path = current_path.parent.parent.parent
os.environ["BRT_CACHE_PATH"] = str(pkg_path / ".cache")

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["hatch.linewidth"] = 2
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["text.usetex"] = True

import numpy as np

from brt.runtime import BRT_CACHE_PATH

DATA_PATH = BRT_CACHE_PATH / "results"
FIGURE_PATH = BRT_CACHE_PATH / "results/figures"
FIGURE_PATH.mkdir(parents=True, exist_ok=True)

convs = dict.fromkeys([0], bytes.decode)
msdnet_e2e_latency_raw = np.genfromtxt(
    DATA_PATH / "msdnet_all_opt.csv",
    delimiter=",",
    dtype=(
        object,
        float,
        float,
        float,
    ),
    converters=convs,
)

# Pre-process
exit_portions = [
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.4, 0.6],
    [0.0, 0.0, 0.3, 0.3, 0.4],
    [0.1, 0.1, 0.2, 0.3, 0.3],
    [0.5, 0.2, 0.2, 0.1, 0.0],
    [0.5, 0.3, 0.2, 0.0, 0.0],
    [0.5, 0.5, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0],
]

msdnet_e2e_latency = []
for i, ep in enumerate(exit_portions):
    start = 800 * i
    vf_mean = msdnet_e2e_latency_raw[start:start+100]["f1"].mean()
    sp_mean = msdnet_e2e_latency_raw[start+600:start+700]["f1"].mean()
    hf_mean = msdnet_e2e_latency_raw[start+700:start+800]["f1"].mean()
    msdnet_e2e_latency.append((str(ep), vf_mean, sp_mean, hf_mean))
msdnet_e2e_latency = np.array(msdnet_e2e_latency, dtype="O,f,f,f")[::-1]
print(msdnet_e2e_latency)

# Draw

t = 0.9  # 1-t == top space
b = 0.1  # bottom space      (both in figure coordinates)
msp = 0.1  # minor spacing
sp = 0.5  # major spacing
hspace = sp + msp + 1  # height space per grid

group_num = 8
group_width = 5
group_start_x = np.arange(group_num) * group_width
bar_num = 3
bar_width = group_width * 0.9 / bar_num

print(group_start_x)
print(bar_width)

edgecolors = ["dimgrey", "lightseagreen", "tomato", "slategray", "silver"]
hatches = ["/////", "\\\\\\\\\\"]
labels = ["BRT+VF", "BRT+SP", "BRT+HF"]
d = 0.015

fig = plt.figure()
fig.set_size_inches(6, 3)

ax = plt.subplot()

for j in range(bar_num):
    ax.barh(
        group_start_x + bar_width * (bar_num - j - 1),
        msdnet_e2e_latency[:][f"f{j + 1}"],
        height=bar_width * 0.8,
        color="white",
        edgecolor=edgecolors[j],
        hatch=hatches[j % 2],
        linewidth=1.0,
        label=labels[j],
        zorder=10,
    )

ax.legend(
    # bbox_to_anchor=(0.98, 1.5),
    ncol=1,
    loc="best",
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
    prop={"weight": "bold", "size": 10},
).set_zorder(100)
ax.set_yticks(group_start_x + group_width / 2 - bar_width / 2)
tick_lables = msdnet_e2e_latency[:]["f0"]
tick_lables = [w.replace(":", ",") for w in tick_lables]
plt.text(-2.1, -3.5, "Exit Portion", fontsize=14, zorder=100)
ax.set_yticklabels(tick_lables, fontsize=14)
ax.set_xlabel("Time (\\textit{ms})", fontsize=14)
ax.xaxis.set_tick_params(labelsize=14)

plt.savefig(FIGURE_PATH / "figure_20.pdf", bbox_inches="tight", dpi=300)
