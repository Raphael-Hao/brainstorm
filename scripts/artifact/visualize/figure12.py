
from matplotlib import pyplot as plt

from brt.runtime import BRT_CACHE_PATH

DATA_PATH = BRT_CACHE_PATH / "results"
FIGURE_PATH = BRT_CACHE_PATH / "results/figures"
FIGURE_PATH.mkdir(parents=True, exist_ok=True)


plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["hatch.linewidth"] = 2
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["text.usetex"] = True

import numpy as np

fusion_latency_raw = np.genfromtxt(
    DATA_PATH / "homo_conv2d.csv",
    delimiter=",",
    dtype=(object, int, float, float, float),
)

fusion_latency = []
for i in range(1, 9):
    fusion_latency.append(
        (2**i, *fusion_latency_raw[fusion_latency_raw["f1"] == 2**i]["f2"])
    )
print(fusion_latency)
fusion_latency = np.array(fusion_latency)

fig, ax = plt.subplots()
fig.set_size_inches(5, 2)
x_ticks = np.log2(fusion_latency[:, 0])
print(fusion_latency[:, 0])
print(x_ticks)
ax.plot(
    x_ticks,
    fusion_latency[:, 1],
    marker="*",
    mfc="white",
    color="dimgrey",
    linewidth=1.5,
    label="Torch",
)
ax.plot(
    x_ticks,
    fusion_latency[:, 2],
    marker="X",
    mfc="white",
    color="tomato",
    linewidth=1.5,
    label="BRT-VF",
)
ax.plot(
    x_ticks,
    fusion_latency[:, 3],
    marker=".",
    mfc="white",
    color="lightseagreen",
    linewidth=1.5,
    label="BRT-HF",
)

ax.set_xticklabels(int(2**e) for e in range(int(fusion_latency[-1, 0])))
ax.set_xlabel("Number of Branches", fontsize=12, fontweight="bold")
# ax.set_xscale("log", base=2)
ax.set_ylabel("Time (\\textit{ms})", fontsize=12, fontweight="bold")
ax.set_yscale("log")

ax.legend(
    ncol=3,
    loc="best",
    markerscale=1,
    labelspacing=0.1,
    edgecolor="black",
    shadow=False,
    fancybox=False,
    handlelength=1.2,
    handletextpad=0.6,
    columnspacing=0.8,
    prop={"weight": "bold", "size": 10},
)
plt.savefig(FIGURE_PATH / "figure_12.pdf", dpi=300, bbox_inches="tight")

