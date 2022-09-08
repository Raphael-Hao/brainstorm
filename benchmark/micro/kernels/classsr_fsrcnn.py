# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.


import json

#%%
from typing import Dict, List, Union, Any

import torch.nn as nn
from brt.runtime import BRT_KERNEL_TEMPLATE_PATH, BRT_LOG_PATH

from brt.jit.tvm import TVMTuner
from brt.jit.tvm.utils import make_fname
from torch.nn.modules.utils import _pair

from brt.runtime import log

# from ...classsr.codes.models.archs.classSR_fsrcnn_arch import classSR_3class_fsrcnn_net

logger = log.get_logger()
logger.setLevel("INFO")


def parse_conv2d_bn_act_params(json_params: Dict[str, Any]):

    if json_params.get("module_name", None) == None:
        json_params["module_name"] = "Conv2d"
    out_channels = json_params["out_channels"]
    json_params["kernel_size"] = _pair(json_params["kernel_size"])
    if json_params["stride"] == None:
        json_params["stride"] = _pair(1)
    else:
        json_params["stride"] = _pair(json_params["stride"])
    if json_params["padding"] == None:
        json_params["padding"] = _pair(0)
    else:
        json_params["padding"] = _pair(json_params["padding"])
    if json_params["dilation"] == None:
        json_params["dilation"] = _pair(1)
    else:
        json_params["dilation"] = _pair(json_params["dilation"])
    if json_params["groups"] == None:
        json_params["groups"] = 1
    bias = json_params.pop("bias")
    if bias is True:
        json_params["module_name"] += "Bias"
    padding_mode = json_params.pop("padding_mode")
    if (
        json_params["module_name"] == "ConvTranspose2d"
        and json_params.get("output_padding", None) == None
    ):
        json_params["output_padding"] = 0
    norm = json_params.pop("norm")
    if norm == "SyncBatchNorm":
        norm = nn.BatchNorm2d(out_channels)
        json_params["module_name"] += "BatchNorm"
    elif norm is not None:
        assert False, f'Unsupported norm type {norm}'
    activation = json_params.pop("activation")
    if activation == "ReLU":
        activation = nn.ReLU()
        json_params["module_name"] += "ReLU"
    elif activation == "PReLU":
        # Note: assert num_parameters == 1
        activation = nn.PReLU()
        json_params["module_name"] += "PReLU"
    elif activation is not None:
        assert False, f'Unsupported activation type {activation}'
    input_infos = {"input_0": json_params.pop("input_shape")}
    output_infos = {"output_0": json_params.pop("output_shape")}
    parameters = json_params
    return (
        json_params["module_name"],
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


def make_log_fname(
    op_type: str,
    method: str,
    input_infos: Dict[str, List[int]],
    output_infos: Dict[str, List[int]],
    parameters: Dict[str, Union[Union[int, float], List[Union[int, float]]]],
) -> str:
    fname = op_type + "_" + method
    fname += "_"
    fname += "-".join(
        f"{name}_" + "_".join(str(dim) for dim in shape)
        for name, shape in input_infos.items()
    )
    fname += "_"
    fname += "_".join(
        f"{name}_" + "_".join(str(dim) for dim in shape)
        for name, shape in output_infos.items()
    )
    fname += "_"
    fname += "_".join(
        f"{name}_" + str(parameter[0])
        if isinstance(parameter, (list, tuple))
        else f"{name}_" + str(parameter)
        for name, parameter in parameters.items()
    )
    return fname


def main():
    tvm_tuner = TVMTuner()
    conv_params_log_file = (
        BRT_LOG_PATH / "benchmark/classsr/fsrcnn/conv_params_AIC21_Track1_Cam1.json"
    )
    conv_params_log_file_nodup = (
        BRT_LOG_PATH / "benchmark/classsr/fsrcnn/conv_params_nodup.json"
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
            if "Conv2d" in module_name:
                operator = Conv2dBNAct(
                    in_channels=parameters["in_channels"],
                    out_channels=parameters["out_channels"],
                    kernel_size=parameters["kernel_size"],
                    stride=parameters["stride"],
                    padding=parameters["padding"],
                    groups=parameters["groups"],
                    bias=bias,
                    dilation=parameters["dilation"],
                    padding_mode=padding_mode,
                    norm=norm,
                    activation=activation,
                )
            elif "ConvTranspose2d" in module_name:
                # Note: unused `norm` and `activation`
                operator = nn.ConvTranspose2d(
                    in_channels=parameters["in_channels"],
                    out_channels=parameters["out_channels"],
                    kernel_size=parameters["kernel_size"],
                    stride=parameters["stride"],
                    padding=parameters["padding"],
                    output_padding=parameters["output_padding"],
                    groups=parameters["groups"],
                    bias=bias,
                    dilation=parameters["dilation"],
                    padding_mode=padding_mode,
                )
            else:
                assert False, f"Unsupport model type {parameters['model_name']}"
            logger.info(parameters)
            tvm_tuner.import_pt_netlet(
                module_name,
                "forward",
                operator,
                input_infos,
                output_infos,
                parameters,
                # make_log_fname(
                #     module_name, "forward", input_infos, output_infos, parameters
                # ),
            )
            logger.info(f"tuning {module_name} with: {parameters}")
            tvm_tuner.tune_netlet()
            tvm_tuner.export_netlet_template()
            tvm_tuner.insert_netlet_to_storage()
            # module_function = ModuleKernel(
            #     module_name,
            #     "forward",
            #     None,
            #     "CUDA_GPU",
            #     input_infos,
            #     output_infos,
            #     parameters,
            # )
            # module_function.load_from_db()
            # file_name = make_fname(
            #     module_name, "forward", input_infos, output_infos, parameters
            # )
            # template_file_loaded = BRT_KERNEL_TEMPLATE_PATH / f"{file_name}_loaded.cu"
            # template_file_loaded.write_text(module_function.get_code()[0])
    print('Done!')


if __name__ == "__main__":
    main()

# %%
