# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
#%%
from typing import List

import torch
import torch.nn as nn
from brt.common import log
from brt.jit import make_jit_kernel

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
        bias,
        padding_mode,
        norm,
        activation,
    )


class Conv2dBNAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        padding_mode,
        norm,
        activation,
    ) -> None:
        super().__init__()
        self.sub_modules = []
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.sub_modules.append(self.conv)
        if norm is None:
            self.norm = None
        else:
            self.norm = norm
            self.sub_modules.append(self.norm)
        if activation is None:
            self.activation = None
        else:
            self.activation = activation
            self.sub_modules.append(self.activation)

    def get_submodues(self):
        return self.sub_modules

    def forward(self, inputs):
        x = self.conv(inputs)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


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

modules = []
for params in conv_params:
    (
        module_name,
        input_infos,
        output_infos,
        parameters,
        bias,
        padding_mode,
        norm,
        activation,
    ) = parse_conv2d_bn_act_params(params)
    conv2d_bn_act = Conv2dBNAct(
        in_channels=parameters["in_channels"],
        out_channels=parameters["out_channels"],
        kernel_size=parameters["kernel_size"],
        stride=parameters["stride"],
        padding=parameters["padding"],
        dilation=parameters["dilation"],
        groups=parameters["groups"],
        bias=bias,
        padding_mode=padding_mode,
        norm=norm,
        activation=activation,
    )
    modules.append(nn.Sequential(*conv2d_bn_act.get_submodues()))
modules = torch.nn.ModuleList(modules).cuda()
sample_inputs = []
data_0 = torch.ones((1, 3, 1024, 2048), device="cuda")
data_1 = torch.ones((1, 64, 511, 1023), device="cuda")
data_2 = torch.ones((1, 64, 511, 1023), device="cuda")
sample_inputs.append(data_0)
sample_inputs.append(data_1)
sample_inputs.append(data_2)

jit_conv_kernel = make_jit_kernel(modules, sample_inputs, opt_level="hetero_fuse")


#%%

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

active_blocks = [1, 1, 1]

jit_conv_kernel(
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
    jit_conv_kernel(
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
