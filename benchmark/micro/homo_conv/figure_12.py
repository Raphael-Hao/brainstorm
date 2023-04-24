# %%
import itertools as itt

import torch
from torch import nn

torch.no_grad()

import brt
from brt.jit import make_jit_kernel, make_jit_module
from brt.runtime.benchmark import CUDATimer

# %%
timer = CUDATimer(export_fname="homo_conv2d")

n_branches = [2**i for i in range(1, 9)]
n_channels = 36
batchsize = 1

conv2d = nn.Sequential(
    nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)
).cuda()
inputs = torch.empty(batchsize, n_channels, 32, 32).cuda()

vf_kernel = make_jit_kernel(conv2d, inputs)

for nb in n_branches:
    conv2ds = [
        nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)
        ).cuda()
        for _ in range(nb)
    ]
    inputs = [torch.empty(batchsize, n_channels, 32, 32).cuda() for _ in range(nb)]
    outputs = [torch.empty(batchsize, n_channels, 32, 32).cuda() for _ in range(nb)]

    timer.execute(
        lambda: [conv2ds[0](inputs[0]) for _ in range(nb)],
        msg=f"Torch,{nb}",
        export=True,
    )

    timer.execute(
        lambda: [
            vf_kernel(x, conv[0].weight, out, conv[0].bias)
            for conv, x, out in zip(conv2ds, inputs, outputs)
        ],
        msg=f"BRT+VF,{nb}",
        export=True,
    )

    hf_kernel = make_jit_kernel(
        nn.ModuleList(conv2ds), sample_inputs=inputs, opt_level="horiz_fuse"
    )
    kernel_args = list(
        itt.chain.from_iterable(
            [i, conv2d[0].weight, o, conv2d[0].bias] for i, o in zip(inputs, outputs)
        )
    )
    timer.execute(
        lambda: hf_kernel(*kernel_args),
        msg=f"BRT+HF,{nb}",
        export=True,
    )

# %%
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
for i in range(1, 7):
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

# %%
