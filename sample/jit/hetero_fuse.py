# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
#%%
from typing import List

import torch
import torch.nn as nn
from brt.common import BRT_KERNEL_TEMPLATE_PATH, log
from brt.jit import CUDACompiler
from brt.jit.function import HeteroFusedFunction, ModuleFunction

log.set_level("jit", "DEBUG")


def parse_conv2d_bn_act_params(json_params):
    module_name = "Conv2d"
    out_channels = json_params["out_channels"]
    if json_params["stride"] == None:
        json_params["stride"] = 1
    if json_params["padding"] == None:
        json_params["padding"] = 0
    if json_params["dilation"] == None:
        json_params["dilation"] = 1
    if json_params["groups"] == None:
        json_params["groups"] = 1
    bias = json_params.pop("bias")
    if bias is True:
        module_name += "Bias"
    padding_mode = json_params.pop("padding_mode")
    norm = json_params.pop("norm")
    if norm == "SyncBatchNorm":
        norm = nn.BatchNorm2d(out_channels)
        module_name += "BatchNorm"
    activation = json_params.pop("activation")
    if activation == "ReLU":
        activation = nn.ReLU()
        module_name += "ReLU"
    input_infos = {"input_0": json_params.pop("input_shape")}
    output_infos = {"output_0": json_params.pop("output_shape")}
    parameters = json_params
    return (
        module_name,
        input_infos,
        output_infos,
        parameters,
        padding_mode,
        norm,
        activation,
    )


conv_params = [
    {
        "in_channels": 3,
        "out_channels": 64,
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "bias": False,
        "padding_mode": "zeros",
        "norm": "SyncBatchNorm",
        "activation": "ReLU",
        "input_shape": [1, 3, 1024, 2048],
        "output_shape": [1, 64, 511, 1023],
    },
    {
        "in_channels": 64,
        "out_channels": 64,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "groups": 64,
        "bias": False,
        "padding_mode": "zeros",
        "norm": None,
        "activation": None,
        "input_shape": [1, 64, 511, 1023],
        "output_shape": [1, 64, 511, 1023],
    },
    {
        "in_channels": 64,
        "out_channels": 64,
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "bias": False,
        "padding_mode": "zeros",
        "norm": "SyncBatchNorm",
        "activation": "ReLU",
        "input_shape": [1, 64, 511, 1023],
        "output_shape": [1, 64, 511, 1023],
    },
]
candidates: List[ModuleFunction] = []
for params in conv_params:
    (
        module_name,
        input_infos,
        output_infos,
        parameters,
        padding_mode,
        norm,
        activation,
    ) = parse_conv2d_bn_act_params(params)
    candidates.append(
        ModuleFunction(
            module_name,
            "forward",
            None,
            "CUDA_GPU",
            input_infos=input_infos,
            output_infos=output_infos,
            parameters=parameters,
        )
    )
    candidates[-1].load_from_db()

module_func = HeteroFusedFunction(candidates)

code, deps, sig, body = module_func.get_code()

processed_template_fname = str(
    BRT_KERNEL_TEMPLATE_PATH / ("processed_" + module_func.module_name + ".cu")
)
with open(processed_template_fname, "w") as f:
    f.write(code)

#%%

fused_conv = CUDACompiler.generate_kernel(None, code)

data_0 = torch.ones((1, 3, 1024, 2048), device="cuda")
weight_0 = torch.ones((3, 64, 3, 3), device="cuda")
outdata_0 = torch.zeros((1, 64, 511, 1023), device="cuda")
bn_0 = torch.ones((64), device="cuda")
data_1 = torch.ones((1, 64, 511, 1023), device="cuda")
weight_1 = torch.ones((64, 1, 3, 3), device="cuda")
outdata_1 = torch.zeros((1, 64, 511, 1023), device="cuda")
data_2 = torch.ones((1, 64, 511, 1023), device="cuda")
weight_2 = torch.ones((64, 64, 1, 1), device="cuda")
outdata_2 = torch.zeros((1, 64, 511, 1023), device="cuda")
bn_2 = torch.ones((64), device="cuda")

torch.cuda.synchronize()
stream = torch.cuda.default_stream()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record(stream)

active_blocks = [1, 0, 0]

fused_conv(
    data_0,
    weight_0,
    outdata_0,
    bn_0,
    data_1,
    weight_1,
    outdata_1,
    data_2,
    weight_2,
    outdata_2,
    bn_2,
    active_blocks=active_blocks,
)

end_event.record(stream)
stream.synchronize()
print("first time: {:.3f}".format(start_event.elapsed_time(end_event)))

start_event.record(stream)
for i in range(100):
    fused_conv(
        data_0,
        weight_0,
        outdata_0,
        bn_0,
        data_1,
        weight_1,
        outdata_1,
        data_2,
        weight_2,
        outdata_2,
        bn_2,
        active_blocks=active_blocks,
    )
end_event.record(stream)
stream.synchronize()
print(outdata_0)
print(outdata_1)
print(outdata_2)
print("forward time: {:.3f}".format(start_event.elapsed_time(end_event) / 100))

# %%
