# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch

from torch.utils.benchmark import Timer


from brt.jit import make_jit_kernel

from brt.jit.tvm import TVMTuner

from brt.jit.codegen import ModuleKernel


bs = 128

in_features = 512

out_features = 1024


input_infos = {"input_0": (bs, in_features)}

output_infos = {"output_0": (bs, out_features)}

parameters = {
    "in_features": in_features,
    "out_features": out_features,
}

kernel_name = f"Linear_{bs}_{in_features}_{out_features}"


linear = torch.nn.Linear(in_features, out_features).eval()


tvm_tuner = TVMTuner()
tvm_tuner.import_pt_netlet(
    "LinearBias",
    "forward",
    linear,
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

    time = (
        Timer(
            stmt="y = torch.empty(oshape).cuda(); model(x, weight, y, bias)",
            setup="import torch",
            globals={
                "model": linear_kernel,
                "x": x,
                "y": y,
                "weight": linear.weight,
                "bias": linear.bias,
                "oshape": (bs, out_features),
            },
        )
        .timeit(1000)
        .mean
        * 1e6
    )

    print(f"{rank = }, {time}")
