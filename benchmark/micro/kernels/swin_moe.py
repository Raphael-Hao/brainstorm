# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.


import torch.nn as nn

from brt.runtime import log, BRT_KERNEL_TEMPLATE_PATH
from brt.jit.codegen import ModuleKernel
from brt.jit.tvm import TVMTuner
from brt.jit.tvm.utils import make_fname

logger = log.get_logger()


class Expert(nn.Module):
    def __init__(self, in_features=512, out_features=1024, bias=True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, inputs):
        return self.linear(inputs)


def main():
    logger.setLevel("INFO")
    input_bs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    in_features_s = [96, 192, 384, 768]  # 512
    out_features_s = [384, 768, 1536, 3072]  # 1024

    all_ignored_features_s = []
    # all_ignored_features_s = [(96, 384), (192, 768), (384, 1536)]
    ignored_set = set()
    for in_features, out_features in all_ignored_features_s:
        for bs in input_bs:
            ignored_set.add((bs, in_features, out_features))
    partial_ignored_features_s = []
    # partial_ignored_features_s = [(768, 3072)]
    partial_ignored_input_bs = [
        1,
        2,
        4,
        8,
        16,
        32,
    ]
    for in_features, out_features in partial_ignored_features_s:
        for bs in partial_ignored_input_bs:
            ignored_set.add((bs, in_features, out_features))
    print("ignored_set:", ignored_set)
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
                if (bs, in_features, out_features) not in ignored_set:
                    print(f"tuning {module_name} with: {parameters}")
                    tvm_tuner.tune_netlet()
                else:
                    print(f"tuning ignored for {module_name} with: {parameters}")
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
                # template_file_loaded = (
                #     BRT_KERNEL_TEMPLATE_PATH / f"{file_name}_loaded.cu"
                # )
                # template_file_loaded.write_text(module_function.get_code()[0])


if __name__ == "__main__":
    main()

# %%
