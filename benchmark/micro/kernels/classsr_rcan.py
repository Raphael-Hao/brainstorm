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

def parse_conv2d_bn_act_params(params: Dict[str, Any]):
    module_name = params.pop("module_name")
    if module_name is None:
        module_name = "Conv2d"
    if module_name in ["Conv2d", "ConvTranspose2d"]:
        # default value
        params["kernel_size"] = _pair(params["kernel_size"])
        if params["stride"] == None:
            params["stride"] = 1
        params["stride"] = _pair(params["stride"])
        if params["padding"] == None:
            params["padding"] = 0
        params["padding"] = _pair(params["padding"])
        if params["dilation"] == None:
            params["dilation"] = 1
        params["dilation"] = _pair(params["dilation"])
        if params["groups"] == None:
            params["groups"] = 1
        if params["bias"] == None:
            params["bias"] = True
        if params["padding_mode"] == None:
            params["padding_mode"] = "zeros"
        if (
            module_name == "ConvTranspose2d"
            and params.get("output_padding", None) == None
        ):
            params["output_padding"] = 0
        # generate module_name & operator
        modules = []
        if module_name == "Conv2d":
            modules.append(
                nn.Conv2d(
                    in_channels = params["in_channels"],
                    out_channels = params["out_channels"],
                    kernel_size = params["kernel_size"],
                    stride = params["stride"],
                    padding = params["padding"],
                    dilation = params["dilation"],
                    groups = params["groups"],
                    bias = params["bias"],
                    padding_mode = params["padding_mode"],
                )
            )
        else:
            modules.append(
                nn.ConvTranspose2d(
                    in_channels = params["in_channels"],
                    out_channels = params["out_channels"],
                    kernel_size = params["kernel_size"],
                    stride = params["stride"],
                    padding = params["padding"],
                    output_padding = params["output_padding"],
                    groups = params["groups"],
                    bias = params["bias"],
                    dilation = params["dilation"],
                    padding_mode = params["padding_mode"],
                )
            )
        bias = params.pop("bias")
        if bias is True:
            module_name += "Bias"
        norm = params.pop("norm")
        if norm == "SyncBatchNorm":
            modules.append(nn.BatchNorm2d(params["out_channels"]))
            module_name += "BatchNorm"
        elif norm is not None:
            assert False, f"Unsupported norm type {norm}"
        activation = params.pop("activation")
        if activation == "ReLU":
            modules.append(nn.ReLU())
            module_name += "ReLU"
        elif activation == "PReLU":
            # Note: assert num_parameters == 1
            modules.append(nn.PReLU())
            module_name += "PReLU"
        elif activation == "Sigmoid":
            # Note: assert num_parameters == 1
            modules.append(nn.Sigmoid())
            module_name += "Sigmoid"
        elif activation is not None:
            assert False, f"Unsupported activation type {activation}"
        # if len(modules) == 1:
        #     operator = modules[0]
        # else:
        operator = nn.Sequential(*modules)
    if module_name == "AdaptiveAvgPool2d":
        operator = nn.AdaptiveAvgPool2d(params["output_size"])
    input_infos = {"input_0": params.pop("input_shape")}
    output_infos = {"output_0": params.pop("output_shape")}
    return (
        module_name,
        input_infos,
        output_infos,
        params,
        operator,
    )


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
        BRT_LOG_PATH / "benchmark/classsr/rcan/conv_params_AIC21_Track1_Cam1.json"
    )
    conv_params_log_file_nodup = (
        BRT_LOG_PATH / "benchmark/classsr/rcan/conv_params_nodup.json"
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
                operator,
            ) = parse_conv2d_bn_act_params(conv_param)
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
    print("Done!")


if __name__ == "__main__":
    main()

# %%
