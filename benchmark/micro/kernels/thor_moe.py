# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.


#%%

import torch
import torch.nn as nn
import tvm
import tvm.auto_scheduler as auto_scheduler
import tvm.relay as relay
from brt.common import BRT_KERNEL_TEMPLATE_PATH, BRT_KERNEL_TUNE_LOG_PATH
from brt.jit.kernel import ModuleKernel
from brt.jit.tvm import TVMTuner
from brt.jit.tvm.utils import (
    make_culaunch_config_str,
    make_fname,
    parse_culaunch_config,
)
from tvm.auto_scheduler.measure_record import load_best_record


class Expert(nn.Module):
    def __init__(self, in_features=512, out_features=1024) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, inputs):
        return self.linear(inputs)


def tvm_tune(model, input_shape, K, N):
    model.eval()
    input_data = torch.randn(input_shape)
    script_model = torch.jit.trace(model, input_data)
    mod, params = relay.frontend.from_pytorch(
        script_module=script_model, input_infos=[("input0", input_data.shape)]
    )
    target = tvm.target.Target("cuda")
    kernel_name = f"Linear_GELU_{input_shape[0]}_{K}_{N}"
    log_file = kernel_name + ".json"
    log_fpath = BRT_KERNEL_TUNE_LOG_PATH / log_file
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    for idx, task in enumerate(tasks):
        print(
            "========== Task %d  (workload key: %s) =========="
            % (idx, task.workload_key)
        )
        print(task.compute_dag)
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(
        repeat=2, min_repeat_ms=300, timeout=10
    )
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=1000,  # change this to 20000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(str(log_fpath))],
    )
    tuner.tune(tune_option)
    if len(tasks) == 1:
        tvm_sch, tvm_args = tasks[0].apply_best(str(log_fpath))
        tvm_ir = tvm.lower(tvm_sch, tvm_args, simple_mode=True)
        grid_dim, block_dim = parse_culaunch_config(tvm_ir)
        culaunch_config = make_culaunch_config_str(grid_dim, block_dim)
        source_code = tasks[0].print_best(str(log_fpath), print_mode="cuda")
        kernel_template = culaunch_config + source_code
        template_fpath = BRT_KERNEL_TEMPLATE_PATH / (kernel_name + ".cu")
        with open(template_fpath, "w") as f:
            f.write(kernel_template)


def main():
    input_bs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    in_features = 1024
    out_features = 512
    # in_features = 512
    # out_features = 1024
    tvm_tuner = TVMTuner()
    for bs in input_bs:
        input_infos = {"input_0": (bs, in_features)}
        output_infos = {"output_0": (bs, out_features)}
        parameters = {
            "in_features": in_features,
            "out_features": out_features,
        }
        kernel_name = f"Linear_{bs}_{in_features}_{out_features}"
        log_fname = kernel_name
        tvm_tuner.import_pt_netlet(
            "LinearBias",
            "forward",
            Expert(in_features, out_features),
            input_infos,
            output_infos,
            parameters,
            # log_fname,
        )
        tvm_tuner.export_netlet_template()
        tvm_tuner.insert_netlet_to_storage()
        # module_function = ModuleKernel(
        #     "LinearBias",
        #     "forward",
        #     None,
        #     "CUDA_GPU",
        #     input_infos,
        #     output_infos,
        #     parameters,
        # )
        # module_function.load_from_db()
        # file_name = make_fname(
        #     "LinearBias", "forward", input_infos, output_infos, parameters
        # )
        # template_file_loaded = BRT_KERNEL_TEMPLATE_PATH / f"{file_name}_loaded.cu"
        # template_file_loaded.write_text(module_function.get_code()[0])


if __name__ == "__main__":
    main()

# %%
