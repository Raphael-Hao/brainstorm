#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /tvm_tune.py
# \brief:
# Author: raphael hao
import onnx
import numpy as np
import tvm
import tvm.relay as relay
from tvm import auto_scheduler
from tvm.contrib import graph_executor

serial_onnx_model = onnx.load("fusion_2_thor_model.onnx")
# fusion_onnx_model = onnx.load("fusion_thor_model.onnx")

serial_mod, serial_params = relay.frontend.from_onnx(serial_onnx_model, opset=11)
# fusion_mod, fusion_params = relay.frontend.from_onnx(fusion_onnx_model)
target = tvm.target.Target("cuda")
dtype = "float32"
log_file = "fusion_2_thor_tune.log"

serial_tasks, serial_task_weights = auto_scheduler.extract_tasks(
    serial_mod["main"], serial_params, target
)

for idx, task in enumerate(serial_tasks):
    print(
        "========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key)
    )
    print(task.compute_dag)


def run_tuning():
    print("Begin tuning...")
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(
        repeat=4, min_repeat_ms=300, timeout=10
    )

    tuner = auto_scheduler.TaskScheduler(serial_tasks, serial_task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=8000,  # change this to 20000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)


run_tuning()

with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(
        opt_level=3, config={"relay.backend.use_auto_scheduler": True}
    ):
        lib = relay.build(serial_mod, target=target, params=serial_params)

# Create graph executor
dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))
data_tvm = tvm.nd.array(
    (np.random.uniform(size=(1, 64, 512))).astype(dtype), device=dev
)
module.set_input("input.1", data_tvm)

# Evaluate
print("Evaluate inference time cost...")
print(module.benchmark(dev, repeat=3, min_repeat_ms=500))
