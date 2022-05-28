# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import json
from typing import Dict, List, Union

import onnx
import torch
from brt.common import (
    BRT_KERNEL_TEMPLATE_PATH,
    BRT_KERNEL_TUNE_LOG_PATH,
    BRT_ONNX_CKPT_PATH,
    log,
)
from brt.jit.module_func import ModuleFunction
from brt.jit.storage import kernel_storager
from brt.jit.utils import make_func_name

import tvm
from tvm import auto_scheduler, relay
from tvm.auto_scheduler.utils import deserialize_args

from .utils import (
    make_culaunch_config_str,
    make_fname,
    make_inputs,
    parse_culaunch_config,
)

logger = log.get_logger(__file__)


class TVMTuner:
    def __init__(
        self,
        dtype="float32",
        platform="CUDA_GPU",
        task_scheduler_cls: auto_scheduler.TaskScheduler = None,
    ) -> None:
        self.dtype = dtype
        self.platform = platform
        if platform == "CUDA_GPU":
            self.target = tvm.target.Target("cuda")
        else:
            raise NotImplementedError
        self.measure_ctx = auto_scheduler.LocalRPCMeasureContext(
            repeat=4, min_repeat_ms=300, timeout=10
        )

        if task_scheduler_cls is None:
            task_scheduler_cls = auto_scheduler.TaskScheduler
        self.task_scheduler_cls = task_scheduler_cls

    def import_pt_netlet(
        self,
        module_name,
        module: torch.nn.Module,
        input_infos: Dict[str, List[Union[int, float]]],
        output_infos: Dict[str, List[Union[int, float]]],
        parameters: Dict[str, List[Union[int, float]]],
        log_fname: str = None,  # for early tuned netlet
    ):
        self.module_name = module_name
        self.input_infos = input_infos
        self.output_infos = output_infos
        self.parameters = parameters
        inputs = make_inputs(input_infos)
        training = module.training
        module.eval()
        tvm_input_infos = [(name, shape) for name, shape in input_infos.items()]
        script_module = torch.jit.trace(module, inputs)
        self.tvm_module, self.tvm_params = relay.frontend.from_pytorch(
            script_module, tvm_input_infos
        )
        self._update_scheduler(
            self.module_name, self.tvm_module, self.tvm_params, log_fname
        )
        module.train(training)

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
            self.module_name = onnx_model
        elif isinstance(onnx_model, onnx.ModelProto):
            onnx_model_proto = onnx_model
            self.module_name = model_name
        self.tvm_module, self.tvm_params = relay.frontend.from_onnx(
            onnx_model_proto, opset=11
        )
        self._update_scheduler(self.module_name, self.tvm_module, self.tvm_params)

    def _update_scheduler(self, module_name, tvm_module, tvm_params, log_fname=None):
        origin_filename = make_fname(
            module_name, self.input_infos, self.output_infos, self.parameters
        )
        filename = make_func_name(
            module_name, self.input_infos, self.output_infos, self.parameters
        )
        if log_fname is None:
            self.old_tune_log_file = (
                BRT_KERNEL_TUNE_LOG_PATH / f"{origin_filename}.json"
            )
            self.tune_log_file = BRT_KERNEL_TUNE_LOG_PATH / f"{filename}.json"
        else:
            self.tune_log_file = BRT_KERNEL_TUNE_LOG_PATH / log_fname
        self.template_file = BRT_KERNEL_TEMPLATE_PATH / f"{filename}.cu"
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

    def get_best_template(self):
        workload = json.loads(self.tvm_task.workload_key)
        kernel_args = deserialize_args(workload[1:])
        logger.debug(f"kernel args: {kernel_args}")
        try:
            if self.old_tune_log_file.exists():
                log_contents = self.old_tune_log_file.read_text()
                self.tune_log_file.write_text(log_contents)
            tvm_sch, tvm_args = self.tvm_task.apply_best(str(self.tune_log_file))
            tvm_ir = tvm.lower(tvm_sch, tvm_args, simple_mode=True)
            grid_dim, block_dim = parse_culaunch_config(tvm_ir)
            source_code = self.tvm_task.print_best(
                self.tune_log_filename, print_mode="cuda"
            )
            return grid_dim, block_dim, source_code
        except Exception as e:
            logger.error(f"Failed to get best netlet {e}")

    def export_netlet_template(self):
        grid_dim, block_dim, source_code = self.get_best_template()
        culaunch_config = make_culaunch_config_str(grid_dim, block_dim)
        kernel_template = culaunch_config + source_code
        with open(self.template_filename, "w") as f:
            f.write(kernel_template)

    def inject_netlet_to_storage(self):
        grid_dim, block_dim, source_code = self.get_best_template()
        culaunch_config = make_culaunch_config_str(grid_dim, block_dim)
        kernel_source = culaunch_config + source_code
        module_function = ModuleFunction(
            self.module_name,
            kernel_source=kernel_source,
            platform=self.platform,
            input_infos=self.input_infos,
            output_infos=self.output_infos,
            parameters=self.parameters,
        )
        module_in_json = module_function.dump_json()
        kernel_storager.add_kernel(module_in_json, overwrite=True)
