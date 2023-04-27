import os
import pathlib

current_path = pathlib.Path(__file__).parent.absolute()
pkg_path = current_path.parent.parent.parent
os.environ["BRT_CACHE_PATH"] = str(pkg_path / ".cache")

from matplotlib import pyplot as plt

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

ALL_NUM_SUBNETS = [10, 8, 6, 4]
ALL_NUM_CHANNELS = [8, 12, 16, 20]

pair_num_subnets_num_channels = []
for num_subnets in ALL_NUM_SUBNETS:
    pair_num_subnets_num_channels.append((num_subnets, ALL_NUM_CHANNELS[0]))
for num_channels in ALL_NUM_CHANNELS[1:]:
    pair_num_subnets_num_channels.append((ALL_NUM_SUBNETS[0], num_channels))

livesr_results_raw = np.genfromtxt(
    DATA_PATH / "benchmark_livesr.csv",
    delimiter=",",
    dtype=(object, float, float, float)

)
livesr_subnet = []
livesr_channel = []
for i, (ns, nc) in enumerate(pair_num_subnets_num_channels):
    if nc == 8:
        livesr_subnet.append((ns, nc, *(livesr_results_raw[j][1] for j in range(3*i, 3*i + 3))))
    if ns == 10:
        livesr_channel.append((ns, nc, *(livesr_results_raw[j][1] for j in range(3*i, 3*i + 3))))

livesr_subnet = np.flip(np.array(livesr_subnet, dtype="i,i,f,f,f"),0)

livesr_channel = np.array(livesr_channel, dtype="i,i,f,f,f")
livesr_e2e = [livesr_subnet, livesr_channel]
print(livesr_e2e)

fig, axes = plt.subplots(1, 2)
fig.set_size_inches(6, 1.5)
plt.subplots_adjust(wspace=0, hspace=0)

total_width, n = 3, 3
group = 4
width = total_width * 0.9 / n
x = np.arange(group) * n
exit_idx_x = x + (total_width - width) / n
edgecolors = ["dimgrey", "lightseagreen", "tomato", "slategray", "silver"]
hatches = ["/////", "\\\\\\\\\\"]
labels = ["BRT", "BRT+VF", "BRT+HF"]
x_labels = ["Number of branches", "Number of channels"]
for i in range(2):
    for j in range(n):
        axes[i].bar(
            exit_idx_x + j * width,
            livesr_e2e[i][:][f"f{j + 2}"],
            width=width * 0.8,
            color="white",
            edgecolor=edgecolors[j],
            hatch=hatches[j % 2],
            linewidth=1.0,
            label=labels[j],
            zorder=10,
        )
    # axes[i].set_yticks([0, 40, 80, 120], fontsize=12)
    axes[i].set_xticks([1.6, 4.6, 7.6, 10.6])
    axes[i].set_xticklabels(livesr_e2e[i][:][f"f{i}"], fontsize=12)
    axes[i].set_xlabel(x_labels[i], fontsize=14)
    axes[i].set_ylim(0,200)
axes[0].legend(
    bbox_to_anchor=(1.5, 0.98),
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
    prop={"weight": "bold", "size": 10},
).set_zorder(100)
axes[0].set_ylabel("Time (\\textit{ms})", fontsize=14)
# axes[0].set_yticks([0, 40, 80, 120], [0, 40, 80, 120], fontsize=12)
# axes[1].set_yticks([0, 40, 80, 120], [], fontsize=12)
axes[1].set_yticklabels([], fontsize=12)
plt.savefig(FIGURE_PATH / "figure16.pdf", bbox_inches="tight", dpi=300)
