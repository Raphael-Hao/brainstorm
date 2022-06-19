# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.


#%%
import json
import logging

import torch.nn as nn
from brt.common import BRT_KERNEL_TEMPLATE_PATH, BRT_LOG_PATH
from brt.jit.function import ModuleFunction
from brt.jit.storage import kernel_storager
from brt.jit.tvm import TVMTuner
from brt.jit.tvm.utils import make_fname

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("brt.microbench.kernels.dynamic_routing")


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
        if norm is None:
            self.norm = None
        else:
            self.norm = norm
        if activation is None:
            self.activation = None
        else:
            self.activation = activation

    def forward(self, inputs):
        x = self.conv(inputs)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def main():
    tvm_tuner = TVMTuner()
    conv_params_log_file = BRT_LOG_PATH / "benchmark/dynamic_routing/conv_params.json"
    conv_params_log_file_nodup = (
        BRT_LOG_PATH / "benchmark/dynamic_routing/conv_params_nodup.json"
    )
    ## remove duplicate lines
    conv_param_set = set()
    nodup_f = conv_params_log_file_nodup.open("w")
    with conv_params_log_file.open("r") as f:
        for line in f.readlines():
            if line not in conv_param_set:
                conv_param_set.add(line)
                nodup_f.write(line)
    nodup_f.close()
    with conv_params_log_file_nodup.open("r") as f:
        for line in f.readlines():
            conv_param = json.loads(line)
            (
                module_name,
                input_infos,
                output_infos,
                parameters,
                bias,
                padding_mode,
                norm,
                activation,
            ) = parse_conv2d_bn_act_params(conv_param)
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
            logger.info(parameters)
            tvm_tuner.import_pt_netlet(
                module_name,
                "forward",
                conv2d_bn_act,
                input_infos,
                output_infos,
                parameters,
            )
            logger.info(f"tuning {module_name} with: {parameters}")
            # tvm_tuner.tune_netlet()
            tvm_tuner.export_netlet_template()
            tvm_tuner.insert_netlet_to_storage()
            module_function = ModuleFunction(
                module_name,
                None,
                "CUDA_GPU",
                input_infos,
                output_infos,
                parameters,
                method="forward",
            )
            module_function.load_from_db()
            file_name = make_fname(
                module_name, "forward", input_infos, output_infos, parameters
            )
            template_file_loaded = BRT_KERNEL_TEMPLATE_PATH / f"{file_name}_loaded.cu"
            template_file_loaded.write_text(module_function.get_code()[0])


if __name__ == "__main__":
    main()

# %%
