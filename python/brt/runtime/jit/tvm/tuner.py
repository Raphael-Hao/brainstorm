# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import json
import pathlib

import onnx
import torch

import tvm
from tvm import auto_scheduler, relay
from tvm.auto_scheduler.utils import deserialize_args

# from brt.common.log import _BRT_PKG_PATH



class TVMTuner:
    def __init__(
        self,
        dtype="float32",
        target="cuda",
        task_scheduler_cls: auto_scheduler.TaskScheduler = None,
        log_dir: str = None,
    ) -> None:
        self.dtype = dtype
        self.raw_target = target
        self.target = tvm.target.Target(self.raw_target)
        self.measure_ctx = auto_scheduler.LocalRPCMeasureContext(
            repeat=4, min_repeat_ms=300, timeout=10
        )

        if task_scheduler_cls is None:
            task_scheduler_cls = auto_scheduler.TaskScheduler
        self.task_scheduler_cls = task_scheduler_cls
        if log_dir is None:
            self.log_dir = pathlib.Path(
                "/home/whcui/brainstorm_project/brainstorm/log"
            ).absolute()
        else:
            self.log_dir = pathlib.Path(log_dir).absolute()

    def import_pt_netlet(self, script_module: torch.ScriptModule, input_infos):
        self.kernel_name = script_module.original_name
        self.tvm_module, self.tvm_params = relay.frontend.from_pytorch(
            script_module, input_infos
        )
        self.log_file = self.log_dir / f"{self.kernel_name}.log"
        self._update_scheduler(self.tvm_module, self.tvm_params)

    def import_onnx_netlet(self, onnx_model: onnx.ModelProto, model_name):
        self.kernel_name = model_name
        self.tvm_module, self.tvm_params = relay.frontend.from_onnx(
            onnx_model, opset=11
        )
        log_file_path = self.log_dir / f"{self.kernel_name}_tune.log"
        self.log_file = str(log_file_path)
        self._update_scheduler(self.tvm_module, self.tvm_params)

    def _update_scheduler(self, tvm_module, tvm_params):
        self.tvm_tasks, self.tvm_task_weights = auto_scheduler.extract_tasks(
            tvm_module["main"], tvm_params, self.target
        )
        # assert (
        #     len(self.tvm_tasks) == 1
        # ), "TVM tuner of BRT only supports one task for horizontal kernel fusion once."
        self.task_scheduler = self.task_scheduler_cls(
            self.tvm_tasks, self.tvm_task_weights
        )
        self.option = auto_scheduler.TuningOptions(
            num_measure_trials=2000,
            runner=self.measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(f"{self.kernel_name}.log")],
        )

    def tune_netlet(self):
        task_scheduler = getattr(self, "task_scheduler", None)
        if task_scheduler is None:
            raise RuntimeError("No netlet imported.")
        task_scheduler.tune(self.option)

    def export_netlet_template(self):
        for idx, task in enumerate(self.tvm_tasks):
            print(
                "========== Task %d  (workload key: %s) =========="
                % (idx, task.workload_key)
            )
            workload = json.loads(task.workload_key)

            if idx == 0:
                source_code = task.print_best(self.log_file, print_mode=self.raw_target)
                kernel_name = self.kernel_name
                kernel_args = deserialize_args(workload[1:])
            print(source_code)
