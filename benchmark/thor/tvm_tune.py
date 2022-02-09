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
import argparse

def run_tuning(tasks, task_weights, log_file):
        print("Begin tuning...")
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(
            repeat=4, min_repeat_ms=300, timeout=10
        )

        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=8000,  # change this to 20000 to achieve the best performance
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )
        tuner.tune(tune_option)


def tune_thor(model_name):
    thor_onnx_model = onnx.load(f"{model_name}.onnx")
    # fusion_onnx_model = onnx.load("fusion_thor_model.onnx")

    thor_mod, thor_params = relay.frontend.from_onnx(thor_onnx_model, opset=11)
    # fusion_mod, fusion_params = relay.frontend.from_onnx(fusion_onnx_model)
    target = tvm.target.Target("cuda")
    dtype = "float32"
    log_file = f"{model_name}_tune.log"

    thor_tasks, thor_task_weights = auto_scheduler.extract_tasks(
        thor_mod["main"], thor_params, target
    )

    for idx, task in enumerate(thor_tasks):
        print(
            "========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key)
        )
        print(task.compute_dag)

    run_tuning(thor_tasks, thor_task_weights, log_file)

    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(
            opt_level=3, config={"relay.backend.use_auto_scheduler": True}
        ):
            lib = relay.build(thor_mod, target=target, params=thor_params)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="thor")
    args = parser.parse_args()
    tune_thor(args.model)