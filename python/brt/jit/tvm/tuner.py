# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import json
from typing import Dict, List, Union

import onnx
import torch
import tvm
from tvm import auto_scheduler, relay

from brt.runtime import log, BRT_KERNEL_TEMPLATE_PATH, BRT_KERNEL_TUNE_LOG_PATH
from brt.jit.codegen import ModuleKernel
from brt.jit.utils import get_device_name
from brt.jit.tvm.utils import (
    make_culaunch_config_str,
    make_fname,
    make_inputs,
    old_make_fname,
    parse_culaunch_config,
)

__all__ = ["TVMTuner"]

logger = log.get_logger(__file__)


class TVMTuner:
    SUPPORTED_OBJECTIVE_FUNCS = ["fastest", "most_efficient"]

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
        method,
        module: torch.nn.Module,
        input_infos: Dict[str, List[Union[int, float]]],
        output_infos: Dict[str, List[Union[int, float]]],
        parameters: Dict[str, Union[Union[int, float], List[Union[int, float]]]],
        log_fname: str = None,  # for early tuned netlet
    ):
        self.module_name = module_name
        self.method = method
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
            self.module_name, self.method, self.tvm_module, self.tvm_params, log_fname
        )
        module.train(training)

    def _update_scheduler(
        self, module_name, method, tvm_module, tvm_params, log_fname: str = None
    ):
        self.filename = make_fname(
            module_name, method, self.input_infos, self.output_infos, self.parameters
        )
        self.device_name = get_device_name(self.platform)
        self.tune_log_file = (
            BRT_KERNEL_TUNE_LOG_PATH / self.device_name / f"{self.filename}.json"
        )
        old_filename = old_make_fname(
            module_name, self.input_infos, self.output_infos, self.parameters
        )
        if log_fname is not None:
            old_filename = log_fname
        self.old_tune_log_file = (
            BRT_KERNEL_TUNE_LOG_PATH / self.device_name / f"{old_filename}.json"
        )
        self.template_file_origin = (
            BRT_KERNEL_TEMPLATE_PATH / self.device_name / f"{self.filename}_origin.cu"
        )
        tvm_tasks, tvm_task_weights = auto_scheduler.extract_tasks(
            tvm_module["main"], tvm_params, self.target
        )
        assert (
            len(tvm_tasks) == 1
        ), "TVM tuner of BRT only supports one task for horizontal kernel fusion once."

        self.tvm_task = tvm_tasks[0]
        self.tvm_task_weight = tvm_task_weights[0]
        self.task_scheduler = self.task_scheduler_cls(
            [self.tvm_task], [self.tvm_task_weight]
        )
        self.option = auto_scheduler.TuningOptions(
            num_measure_trials=1,
            runner=self.measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(str(self.tune_log_file))],
        )

    def tune_netlet(self):
        task_scheduler = getattr(self, "task_scheduler", None)
        if task_scheduler is None:
            raise RuntimeError("No netlet imported.")
        task_scheduler.tune(self.option)

    def get_template_file_generated(self, objective_func: str, rank: int = 1):
        assert (
            objective_func in TVMTuner.SUPPORTED_OBJECTIVE_FUNCS
        ), f"Unsupported {objective_func = }"
        return (
            BRT_KERNEL_TEMPLATE_PATH
            / self.device_name
            / f"{self.filename}_{objective_func}.cu"
        )

    def export_netlet_template(self, objective_func: str = "fastest", rank: int = 1):
        assert (
            objective_func == "all"
            or objective_func in TVMTuner.SUPPORTED_OBJECTIVE_FUNCS
        ), f"Unsupported {objective_func = }"
        if objective_func == "all":
            for f in TVMTuner.SUPPORTED_OBJECTIVE_FUNCS:
                self.export_netlet_template(f, rank)
            return
        kernel_source = self._get_best_source(objective_func, rank)
        module_function = ModuleKernel(
            self.module_name,
            self.method,
            kernel_source=kernel_source,
            platform=self.platform,
            input_infos=self.input_infos,
            output_infos=self.output_infos,
            parameters=self.parameters,
        )
        template_file_generated = self.get_template_file_generated(objective_func, rank)
        if self.template_file_origin.exists():
            logger.warn(f"{self.template_file_origin} already exists.")
        if template_file_generated.exists():
            logger.warn(f"{template_file_generated} already exists.")
        self.template_file_origin.parent.mkdir(parents=True, exist_ok=True)
        self.template_file_origin.write_text(kernel_source)
        template_file_generated.parent.mkdir(parents=True, exist_ok=True)
        template_file_generated.write_text(module_function.get_code()[0])

    def insert_netlet_to_storage(self, objective_func: str = "fastest", rank: int = 1):
        assert (
            objective_func == "all"
            or objective_func in TVMTuner.SUPPORTED_OBJECTIVE_FUNCS
        ), f"Unsupported {objective_func = }"
        if objective_func == "all":
            for f in TVMTuner.SUPPORTED_OBJECTIVE_FUNCS:
                self.insert_netlet_to_storage(f, rank)
            return
        kernel_source = self._get_best_source(objective_func)
        module_function = ModuleKernel(
            self.module_name,
            method=self.method,
            kernel_source=kernel_source,
            platform=self.platform,
            input_infos=self.input_infos,
            output_infos=self.output_infos,
            parameters=self.parameters,
        )
        module_function.dump_to_db(objective_func, rank)

    def _get_best_source(self, objective_func: str, rank: int = 1):
        # if objective_func == "fastest":
        #     grid_dim, block_dim, source_code = self._get_fastest_template()
        # elif objective_func == "most_efficient":
        #     grid_dim, block_dim, source_code = self._get_most_efficient_template()
        grid_dim, block_dim, source_code = self._get_best_template(objective_func)
        culaunch_config = make_culaunch_config_str(grid_dim, block_dim)
        kernel_source = culaunch_config + source_code
        return kernel_source

    def _get_best_template(self, objective_func: str, rank: int = 1):
        # TODO: get the rank-th best template
        if not self.tune_log_file.exists() and self.old_tune_log_file.exists():
            contents = self.old_tune_log_file.read_text()
            self.tune_log_file.parent.mkdir(parents=True, exist_ok=True)
            self.tune_log_file.write_text(contents)
        record_reader = auto_scheduler.RecordReader(str(self.tune_log_file))
        best_score = float("inf")
        best_inp = None
        for inp, res in record_reader:
            if objective_func == "fastest":
                score = sum(res.costs) / len(res.costs)
            elif objective_func == "most_efficient":
                tvm_sch, tvm_args = self.tvm_task.compute_dag.apply_steps_from_state(
                    inp.state
                )
                tvm_ir = tvm.lower(tvm_sch, tvm_args, simple_mode=True)
                grid_dim, block_dim = parse_culaunch_config(tvm_ir)
                num_blocks = grid_dim[0] * grid_dim[1] * grid_dim[2]
                score = num_blocks * sum(res.costs) / len(res.costs)
            else:
                raise ValueError(f"Unsupported {objective_func = }")
            if score < best_score:
                best_score = score
                best_inp = inp
        if best_inp == None:
            raise ValueError(f"Error around record file {str(self.tune_log_file)}")
        best_sch, best_args = self.tvm_task.compute_dag.apply_steps_from_state(
            best_inp.state, self.tvm_task.layout_rewrite_option
        )
        best_ir = tvm.lower(best_sch, best_args, simple_mode=True)
        best_grid_dim, best_block_dim = parse_culaunch_config(best_ir)
        best_func = tvm.driver.build_module.build(best_sch, best_args, "cuda")
        best_source_code = best_func.imported_modules[0].get_source()
        return best_grid_dim, best_block_dim, best_source_code
