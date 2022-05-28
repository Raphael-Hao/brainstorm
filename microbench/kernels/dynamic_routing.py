# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.


#%%
import json
import logging

import torch.nn as nn
from brt.common import BRT_LOG_PATH
from brt.jit.tvm import TVMTuner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("brt.microbench.kernels.dynamic_routing")


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
        BRT_LOG_PATH / "benchmark/benchdynamic_routing/conv_params_nodup.json"
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
            module_name = "Conv2d"
            in_channels = conv_param["in_channels"]
            out_channels = conv_param["out_channels"]
            kernel_size = conv_param["kernel_size"]
            if conv_param["stride"] == None:
                conv_param["stride"] = 1
            stride = conv_param["stride"]
            if conv_param["padding"] == None:
                conv_param["padding"] = 0
            padding = conv_param["padding"]
            if conv_param["dilation"] == None:
                conv_param["dilation"] = 1
            dilation = conv_param["dilation"]
            if conv_param["groups"] == None:
                conv_param["groups"] = 1
            groups = conv_param["groups"]
            bias = conv_param.pop("bias")
            if bias is True:
                module_name += "Bias"
            padding_mode = conv_param.pop("padding_mode")
            norm = conv_param.pop("norm")
            if norm == "SyncBatchNorm":
                norm = nn.BatchNorm2d(out_channels)
                module_name += "BatchNorm"
            activation = conv_param.pop("activation")
            if activation == "ReLU":
                activation = nn.ReLU()
                module_name += "ReLU"
            conv2d_bn_act = Conv2dBNAct(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
                norm=norm,
                activation=activation,
            )
            input_infos = {"input_0": conv_param.pop("input_shape")}
            output_infos = {"output_0": conv_param.pop("output_shape")}
            parameters = conv_param
            logger.info(conv_param)
            tvm_tuner.import_pt_netlet(
                module_name, conv2d_bn_act, input_infos, output_infos, parameters
            )
            logger.info(f"tuning {module_name} with: {parameters}")
            # tvm_tuner.tune_netlet()
            tvm_tuner.export_netlet_template()
            tvm_tuner.insert_netlet_to_storage()


if __name__ == "__main__":
    main()

# %%
