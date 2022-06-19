# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.


#%%

import torch
import torch.nn as nn
import tvm
import tvm.auto_scheduler as auto_scheduler
import tvm.relay as relay
from brt.common import BRT_KERNEL_TEMPLATE_PATH, BRT_KERNEL_TUNE_LOG_PATH, log
from brt.jit.function import ModuleFunction
from brt.jit.tvm import TVMTuner
from brt.jit.tvm.utils import (
    make_culaunch_config_str,
    make_fname,
    parse_culaunch_config,
)
from tvm.auto_scheduler.measure_record import load_best_record

logger = log.get_logger()


class Expert(nn.Module):
    def __init__(self, in_features=512, out_features=1024, bias=True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, inputs):
        return self.linear(inputs)


def main():
    input_bs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    in_features_s = [96, 192, 384, 768]  # 512
    out_features_s = [384, 768, 1536, 3072]  # 1024
    tvm_tuner = TVMTuner()
    for i in range(2):
        bias = True if i == 0 else False
        in_features_iterate = in_features_s if i == 0 else out_features_s
        out_features_iterate = out_features_s if i == 0 else in_features_s
        for in_features, out_features in zip(in_features_iterate, out_features_iterate):
            for bs in input_bs:
                input_infos = {"input_0": (bs, in_features)}
                output_infos = {"output_0": (bs, out_features)}
                parameters = {
                    "in_features": in_features,
                    "out_features": out_features,
                }
                module_name = "Linear" + ("Bias" if bias else "")
                tvm_tuner.import_pt_netlet(
                    module_name,
                    "forward",
                    Expert(in_features, out_features, bias),
                    input_infos,
                    output_infos,
                    parameters,
                )
                print(f"tuning {module_name} with: {parameters}")
                tvm_tuner.tune_netlet()
                tvm_tuner.export_netlet_template()
                tvm_tuner.insert_netlet_to_storage()
                module_function = ModuleFunction(
                    module_name,
                    "forward",
                    None,
                    "CUDA_GPU",
                    input_infos,
                    output_infos,
                    parameters,
                )
                module_function.load_from_db()
                file_name = make_fname(
                    module_name, "forward", input_infos, output_infos, parameters
                )
                template_file_loaded = BRT_KERNEL_TEMPLATE_PATH / f"{file_name}_loaded.cu"
                template_file_loaded.write_text(module_function.get_code()[0])
    # turn off bias


if __name__ == "__main__":
    main()

# %%
