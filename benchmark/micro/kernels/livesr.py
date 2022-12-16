# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import json
from typing import Dict, List, Union, Any

import torch
import torch.nn as nn

from brt.runtime import BRT_KERNEL_TEMPLATE_PATH, BRT_LOG_PATH
from brt.jit.tvm import TVMTuner
from brt.jit.tvm.utils import make_fname
from brt.jit.codegen import ModuleKernel
from brt.runtime import log

from param_parser import parse_params

logger = log.get_logger()
logger.setLevel("INFO")

all_subnet_bs = [
    [6, 7, 12, 27, 8, 8, 8, 12, 12, 4],
    [21, 8, 16, 11, 20, 8, 7, 15],
    [9, 32, 18, 16, 18, 7],
    [19, 25, 18, 36],
]
all_num_channels = [
    8, 12, 16, 20
]


def conv_params(subnet_bs, num_channels):
    unique_bs = sorted(set(subnet_bs))
    return (
        [
            f"""{{"module_name": "Conv2dMulAdd", "in_channels": {num_channels}, "out_channels": {num_channels}, "kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "groups": 1, "bias": true, "padding_mode": "zeros", "norm": null, "activation": null, "input_shape": [{bs}, {num_channels}, 32, 32], "output_shape": [{bs}, {num_channels}, 32, 32]}}\n"""
            for bs in unique_bs
        ]
        + [
            f"""{{"module_name": "Conv2d", "in_channels": {num_channels}, "out_channels": {num_channels}, "kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "groups": 1, "bias": true, "padding_mode": "zeros", "norm": null, "activation": null, "input_shape": [{bs}, {num_channels}, 32, 32], "output_shape": [{bs}, {num_channels}, 32, 32]}}\n"""
            for bs in unique_bs
        ]
        + [
            f"""{{"module_name": "Conv2d", "in_channels": {num_channels}, "out_channels": {num_channels}, "kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "groups": 1, "bias": true, "padding_mode": "zeros", "norm": null, "activation": "ReLU", "input_shape": [{bs}, {num_channels}, 32, 32], "output_shape": [{bs}, {num_channels}, 32, 32]}}\n"""
            for bs in unique_bs
        ]
    )


class Conv2dMulAdd(nn.Module):
    def __init__(
        self,
        conv2d,
        scale=1,
    ) -> None:
        super().__init__()
        self.conv2d = conv2d
        self.scale = scale

    def forward(self, x: torch.Tensor, add: torch.Tensor):
        x = self.conv2d(x)
        x = x * self.scale
        x = x + add
        return x


def tune(subnet_bs, num_channels):
    tvm_tuner = TVMTuner()
    for line in conv_params(subnet_bs, num_channels):
        conv_param = json.loads(line)
        (
            module_name,
            input_infos,
            output_infos,
            parameters,
            operator,
        ) = parse_params(conv_param)

        if module_name == "Conv2dBiasMulAdd":
            operator = Conv2dMulAdd(
                operator,
                scale=1,
            )
            input_infos["input_1"] = input_infos["input_0"]

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

        # try:
        #     tvm_tuner.export_netlet_template("all")
        #     tvm_tuner.insert_netlet_to_storage("all")
        # except ValueError:
        #     break

        module_function = ModuleKernel(
            module_name,
            "forward",
            None,
            "CUDA_GPU",
            input_infos,
            output_infos,
            parameters,
        )
        try:
            # module_function.load_from_db("most_efficient")
            module_function.load_from_db("fastest")
        except ValueError:
            module_function = None

        # # Check tuning times
        # if tvm_tuner.tune_log_file.exists():
        #     with open(tvm_tuner.tune_log_file) as ff:
        #         len_lines = len(ff.readlines())
        #         if len_lines != 2048:
        #             print(f"* {module_name}$ {input_infos['input_0']} -> {output_infos['output_0']}:\t{len_lines}")
        #             # print(f"    {tvm_tuner.tune_log_file}")
        #             module_function = None
        #         else:
        #             continue
        # else:
        #     print(f"* {module_name}$ {input_infos['input_0']} -> {output_infos['output_0']} log file not exist")
        #     # print(f"    {tvm_tuner.tune_log_file}")
        #     module_function = None

        # if True:
        if module_function is not None:
            logger.info("##########################################################")
            logger.info(
                f"####### find tuned {module_name =} with: {parameters =}, {input_infos =}, continue"
            )
            logger.info("##########################################################")
            # tvm_tuner.insert_top_n_netlet_to_storage("fastest")
            # tvm_tuner.insert_top_n_netlet_to_storage("most_efficient", 5)
        else:
            logger.info("##########################################################")
            logger.info(
                f"####### tuning {module_name =} with: {parameters =}, {input_infos =}"
            )
            logger.info("##########################################################")
            tvm_tuner.tune_netlet()
            tvm_tuner.export_netlet_template("fastest")
            tvm_tuner.insert_top_n_netlet_to_storage("fastest")
            # tvm_tuner.insert_top_n_netlet_to_storage("most_efficient", 5)

    print("Done!")


def update_db(subnet_bs, num_channels):
    tvm_tuner = TVMTuner()
    for line in conv_params(subnet_bs, num_channels):
        conv_param = json.loads(line)
        (
            module_name,
            input_infos,
            output_infos,
            parameters,
            operator,
        ) = parse_params(conv_param)

        if module_name == "Conv2dBiasMulAdd":
            operator = Conv2dMulAdd(
                operator,
                scale=1,
            )
            input_infos["input_1"] = input_infos["input_0"]

        logger.info(parameters)
        tvm_tuner.import_pt_netlet(
            module_name,
            "forward",
            operator,
            input_infos,
            output_infos,
            parameters,
        )
        print(f"#### {module_name =} with: {parameters =}, {input_infos =}")
        module_function = ModuleKernel(
            module_name,
            "forward",
            None,
            "CUDA_GPU",
            input_infos,
            output_infos,
            parameters,
        )
        try:
            # module_function.load_from_db("most_efficient")
            module_function.load_from_db("fastest")
            print("     find kernel in db")
        except ValueError:
            module_function = None
        if module_function is None:
            try:
                tvm_tuner.insert_top_n_netlet_to_storage("fastest")
                print("     insert kernel into db")
            except:
                print(f"     can't find kernel")

if __name__ == "__main__":
    for subnet_bs in all_subnet_bs:
        # update_db(subnet_bs, all_num_channels[0])
        tune(subnet_bs, all_num_channels[0])
    for num_channels in all_num_channels:
        # update_db(all_subnet_bs[0], num_channels)
        tune(all_subnet_bs[0], num_channels)
