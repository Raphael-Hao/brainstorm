# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import os

os.environ[
    "BRT_CACHE_PATH"
] = "/home/v-weihaocui/brainstorm_project/brainstorm_raphael/.cache"
import torch
from brt.jit import make_jit_kernel
from brt.jit.codegen import ModuleKernel  # pylint: disable=unused-import

# from brt.runtime import BRT_CACHE_PATH
from brt.jit.tvm import TVMTuner
from torch.utils.benchmark import Timer

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

all_bsz = [
    # 1, # 1,
    2,  # 1,
    4,  # 1,
    8,  # 1,
    16,  # 1,
    32,  # 1,
    64,  # 1,
    96,  # 1,
    128,  # 1,
    160,  # 1,
    192,  # 1,
    224,  # 1,
    272,  # 1,
    320,  # 1, no need may
    368,  # 1,
    416,  # 1,
    512,  # 3,
]

in_features = 768
out_features = 3072
# in_features = 3072
# out_features = 768

for bs in all_bsz:
    input_infos = {"input_0": (bs, in_features)}

    output_infos = {"output_0": (bs, out_features)}

    parameters = {
        "in_features": in_features,
        "out_features": out_features,
    }

    kernel_name = f"Linear_{bs}_{in_features}_{out_features}"

    linear = torch.nn.Linear(in_features, out_features, bias=False).eval().cuda()
    torch_x = torch.randn((bs, in_features)).cuda()
    time = (
        Timer(
            stmt="model(x)",
            setup="import torch",
            globals={
                "model": linear,
                "x": torch_x,
            },
        )
        .timeit(1000)
        .mean
        * 1e6
    )
    print(f"{bs = }, {time}")
    # continue

    tvm_tuner = TVMTuner()
    tvm_tuner.import_pt_netlet(
        "Linear",
        "forward",
        linear.cpu(),
        input_infos,
        output_infos,
        parameters,
        # log_fname,
    )

    tvm_tuner.insert_top_n_netlet_to_storage(top=10)

    linear.cuda()

    for rank in range(1, 11):

        x = torch.randn((bs, in_features)).cuda()

        y = torch.randn((bs, out_features)).cuda()

        linear_kernel = make_jit_kernel(
            modules=linear,
            sample_inputs=x,
            rank=rank,
        )


        y = torch.empty(bs, out_features).cuda(); linear_kernel(x, linear.weight, y)

        time = (
            Timer(
                stmt="y = torch.empty(oshape).cuda(); model(x, weight, y)",
                setup="import torch",
                globals={
                    "model": linear_kernel,
                    "x": x,
                    "y": y,
                    "weight": linear.weight,
                    "oshape": (bs, out_features),
                },
            )
            .timeit(1000)
            .mean
            * 1e6
        )
        print(f"{rank = }, {time}")
