from typing import Dict, List, Any, Union

import torch.nn as nn
from torch.nn.modules.utils import _pair


def parse_params(params: Dict[str, Any]):
    module_name = params.pop("module_name")
    if module_name is None:
        module_name = "Conv2d"
    if module_name in ["Conv2d", "ConvTranspose2d", "Conv2dMulAdd"]:
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
        if module_name in ["Conv2d", "Conv2dMulAdd"]:
            modules.append(
                nn.Conv2d(
                    in_channels=params["in_channels"],
                    out_channels=params["out_channels"],
                    kernel_size=params["kernel_size"],
                    stride=params["stride"],
                    padding=params["padding"],
                    dilation=params["dilation"],
                    groups=params["groups"],
                    bias=params["bias"],
                    padding_mode=params["padding_mode"],
                )
            )
            params.pop("padding_mode")
        elif module_name == "ConvTranspose2d":
            modules.append(
                nn.ConvTranspose2d(
                    in_channels=params["in_channels"],
                    out_channels=params["out_channels"],
                    kernel_size=params["kernel_size"],
                    stride=params["stride"],
                    padding=params["padding"],
                    output_padding=params["output_padding"],
                    groups=params["groups"],
                    bias=params["bias"],
                    dilation=params["dilation"],
                    padding_mode=params["padding_mode"],
                )
            )
        bias = params.pop("bias")
        if bias is True:
            module_name += "Bias"
            if module_name == "Conv2dMulAddBias":
                module_name = "Conv2dBiasAdd"
        norm = params.pop("norm")
        if norm == "SyncBatchNorm":
            modules.append(nn.BatchNorm2d(params["out_channels"]))
            module_name += "BatchNorm2d"
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
    elif module_name == "AdaptiveAvgPool2d":
        operator = nn.AdaptiveAvgPool2d(params["output_size"])
    elif module_name == "MaxPool2d":
        operator = nn.MaxPool2d(params["kernel_size"],params["stride"],params["padding"])
    elif module_name == "AvgPool2d":
        operator = nn.AvgPool2d(params["kernel_size"])
    else:
        import pdb; pdb.set_trace()
        raise ValueError(f"Unsupported module type: {module_name}")
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
