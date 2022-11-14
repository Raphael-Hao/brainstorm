# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import json
from typing import Dict, List, Union, Any

import torch.nn as nn

from brt.runtime import BRT_KERNEL_TEMPLATE_PATH, BRT_LOG_PATH
from brt.jit.tvm import TVMTuner
from brt.jit.tvm.utils import make_fname
from brt.jit.codegen import ModuleKernel
from brt.runtime import log

from param_parser import parse_params

logger = log.get_logger()
logger.setLevel("INFO")


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
            ) = parse_params(conv_param)
            # logger.info(parameters)
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
                module_function.load_from_db("most_efficient")
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

            if module_function is not None:
                logger.info(
                    "##########################################################"
                )
                logger.info(
                    f"####### find tuned {module_name =} with: {parameters =}, {input_infos =}, continue"
                )
                logger.info(
                    "##########################################################"
                )

            else:
                logger.info(
                    "##########################################################"
                )
                logger.info(
                    f"####### tuning {module_name =} with: {parameters =}, {input_infos =}"
                )
                logger.info(
                    "##########################################################"
                )
                tvm_tuner.tune_netlet()
                tvm_tuner.export_netlet_template("all")
                tvm_tuner.insert_netlet_to_storage("all")

            # module_function = ModuleKernel(
            #     module_name,
            #     "forward",
            #     None,
            #     "CUDA_GPU",
            #     input_infos,
            #     output_infos,
            #     parameters,
            # )
            # module_function.load_from_db("most_efficient")

            # file_name = make_fname(
            #     module_name, "forward", input_infos, output_infos, parameters
            # )
            # template_file_loaded = BRT_KERNEL_TEMPLATE_PATH / f"{file_name}_loaded.cu"
            # template_file_loaded.write_text(module_function.get_code()[0])
    print("Done!")


if __name__ == "__main__":
    main()
