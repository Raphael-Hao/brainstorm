import itertools as itt

import torch
from torch import nn

torch.no_grad()

import brt
from brt.jit import make_jit_kernel, make_jit_module
from brt.runtime.benchmark import CUDATimer

timer = CUDATimer(export_fname="homo_conv2d", clean_export=True)

n_branches = [2**i for i in range(1, 9)]
n_channels = 8
batchsize = 4

conv2d = nn.Sequential(
    nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1), nn.ReLU()
).cuda()
inputs = torch.empty(batchsize, n_channels, 32, 32).cuda()

vf_kernel = make_jit_kernel(conv2d, inputs)

for nb in n_branches:
    conv2ds = [
        nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1), nn.ReLU()
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

    hf_kernel = make_jit_kernel(nn.ModuleList(conv2ds), sample_inputs=[inputs[0]], opt_level="homo_fuse")
    shared_args = [torch.empty(nb * batchsize, n_channels, 32, 32).cuda() for _ in range(2)]
    standalone_args = list(itt.chain.from_iterable([conv[0].weight, conv[0].bias] for conv in conv2ds))
    cap = torch.ones(nb, dtype=torch.int32).cuda() * batchsize
    timer.execute(
        lambda: hf_kernel(shared_args, standalone_args, cap),
        msg=f"BRT+HF,{nb}",
        export=True
    )
