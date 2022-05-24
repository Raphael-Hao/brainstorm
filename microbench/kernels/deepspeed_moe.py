# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.


#%%
import argparse
import itertools
import os

import numpy as np
import torch
import torch.nn as nn
import tvm
import tvm.auto_scheduler as auto_scheduler
import tvm.relay as relay
from brt.common import BRT_KERNEL_TEMPLATE_PATH, BRT_KERNEL_TUNE_LOG_PATH
from brt.jit.tvm.utils import make_culaunch_config_str, parse_culaunch_config
from tvm.auto_scheduler.measure_record import load_best_record


class Expert1(nn.Module):
    def __init__(self, hidden_dimension=1024) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dimension, hidden_dimension)

    def forward(self, inputs):
        x = self.linear(inputs)
        return x


def tvm_tune(model, input_shape, K, N):
    model.eval()
    input_data = torch.randn(input_shape)
    script_model = torch.jit.trace(model, input_data)
    mod, params = relay.frontend.from_pytorch(
        script_module=script_model, input_infos=[("input0", input_data.shape)]
    )
    target = tvm.target.Target("cuda")
    kernel_name = f"Linear_{input_shape[0]}_{K}_{N}"
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


def tvm_export(model, input_shape, K, N):
    model.eval()
    input_data = torch.randn(input_shape)
    script_model = torch.jit.trace(model, input_data)
    mod, params = relay.frontend.from_pytorch(
        script_module=script_model, input_infos=[("input0", input_data.shape)]
    )
    target = tvm.target.Target("cuda")
    log_kernel_name = f"matmul_{input_shape[0]}_{K}_{N}"
    template_kernel_name = f"matmul_{K}_{N}_{input_shape[0]}"
    log_file = log_kernel_name + ".json"
    log_fpath = BRT_KERNEL_TUNE_LOG_PATH / log_file
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    for idx, task in enumerate(tasks):
        print(
            "========== Task %d  (workload key: %s) =========="
            % (idx, task.workload_key)
        )
        print(task.compute_dag)
    assert len(tasks) == 1
    tvm_sch, tvm_args = tasks[0].apply_best(str(log_fpath))
    tvm_ir = tvm.lower(tvm_sch, tvm_args, simple_mode=True)
    grid_dim, block_dim = parse_culaunch_config(tvm_ir)
    culaunch_config = make_culaunch_config_str(grid_dim, block_dim)
    source_code = tasks[0].print_best(str(log_fpath), print_mode="cuda")
    kernel_template = culaunch_config + source_code
    template_fpath = BRT_KERNEL_TEMPLATE_PATH / (template_kernel_name + ".cu")
    with open(template_fpath, "w") as f:
        f.write(kernel_template)


def main():
    input_bs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    input_shapes = [[bs, 2048] for bs in input_bs]
    for input_shape in input_shapes:
        tvm_tune(Expert1(2048), input_shape, 2048, 2048)
        # tvm_export(Expert1(), input_shape, 512, 1024)
    # input_bs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    # input_shapes = [[bs, 1024] for bs in input_bs]
    # for input_shape in input_shapes:
    #     tvm_tune(Expert2(), input_shape, 1024, 512)
    #     # tvm_export(Expert2(), input_shape, 1024, 512)


if __name__ == "__main__":
    main()