# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import json
import pathlib
from typing import Union

import onnx
import torch
from brt.common import (
    BRT_KERNEL_TEMPLATE_PATH,
    BRT_KERNEL_TUNE_LOG_PATH,
    BRT_ONNX_CKPT_PATH,
    log,
)

import tvm
from tvm import auto_scheduler, relay
from tvm.auto_scheduler.utils import deserialize_args

from .utils import get_culaunch_config

logger = log.get_logger(__file__)


class TVMTuner:
    def __init__(
        self,
        dtype="float32",
        target="cuda",
        task_scheduler_cls: auto_scheduler.TaskScheduler = None,
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

    def import_pt_netlet(self, script_module: torch.ScriptModule, input_infos):
        self.kernel_name = script_module.original_name
        self.tvm_module, self.tvm_params = relay.frontend.from_pytorch(
            script_module, input_infos
        )
        self._update_scheduler(self.kernel_name, self.tvm_module, self.tvm_params)

    def import_onnx_netlet(
        self, onnx_model: Union[onnx.ModelProto, str], model_name: str = None
    ):
        assert (
            isinstance(onnx_model, str) or model_name is not None
        ), "model_name is required for already loaded onnx model"
        if isinstance(onnx_model, str):
            onnx_model_path = BRT_ONNX_CKPT_PATH / (onnx_model + ".onnx")
            if not onnx_model_path.exists():
                logger.error(f"ONNX model {onnx_model_path} not found.")
            onnx_model_proto = onnx.load(onnx_model_path)
            self.kernel_name = onnx_model
        elif isinstance(onnx_model, onnx.ModelProto):
            onnx_model_proto = onnx_model
            self.kernel_name = model_name
        self.tvm_module, self.tvm_params = relay.frontend.from_onnx(
            onnx_model_proto, opset=11
        )
        self._update_scheduler(self.kernel_name, self.tvm_module, self.tvm_params)

    def _update_scheduler(self, kernel_name, tvm_module, tvm_params):
        self.tune_log_filename = str(BRT_KERNEL_TUNE_LOG_PATH / f"{kernel_name}.log")
        self.template_filename = str(BRT_KERNEL_TEMPLATE_PATH / f"{kernel_name}.cu")
        tvm_tasks, tvm_task_weights = auto_scheduler.extract_tasks(
            tvm_module["main"], tvm_params, self.target
        )
        if len(tvm_tasks) > 1:
            logger.warning(
                "TVM tuner of BRT only supports one task for horizontal kernel fusion once. We will use the first task for tuning."
            )
        self.tvm_task = tvm_tasks[0]
        self.tvm_task_weight = tvm_task_weights[0]
        self.task_scheduler = self.task_scheduler_cls(
            [self.tvm_task], [self.tvm_task_weight]
        )
        self.option = auto_scheduler.TuningOptions(
            num_measure_trials=2000,
            runner=self.measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(self.tune_log_filename)],
        )

    def tune_netlet(self):
        task_scheduler = getattr(self, "task_scheduler", None)
        if task_scheduler is None:
            raise RuntimeError("No netlet imported.")
        task_scheduler.tune(self.option)

    def export_netlet_template(self):
        workload = json.loads(self.tvm_task.workload_key)
        kernel_args = deserialize_args(workload[1:])
        logger.debug(f"kernel args: {kernel_args}")
        tvm_sch, tvm_args = self.tvm_task.apply_best(self.tune_log_filename)
        tvm_ir = tvm.lower(tvm_sch, tvm_args, simple_mode=True)
        culaunch_config = get_culaunch_config(tvm_ir)
        source_code = self.tvm_task.print_best(
            self.tune_log_filename, print_mode="cuda"
        )
        kernel_template = culaunch_config + source_code
        with open(self.template_filename, "w") as f:
            f.write(kernel_template)
