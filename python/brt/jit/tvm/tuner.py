# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import json
from typing import Dict, List, Union
from itertools import groupby

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

        self.tvm_task: auto_scheduler.SearchTask = tvm_tasks[0]
        self.tvm_task_weight = tvm_task_weights[0]
        self.task_scheduler = self.task_scheduler_cls(
            [self.tvm_task], [self.tvm_task_weight]
        )
        self.option = auto_scheduler.TuningOptions(
            num_measure_trials=2000,
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
        kernel_source = self._get_best_source(objective_func, rank)
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

    def insert_top_n_netlet_to_storage(
        self, objective_func: str = "fastest", top: int = 5
    ):
        assert (
            objective_func == "all"
            or objective_func in TVMTuner.SUPPORTED_OBJECTIVE_FUNCS
        ), f"Unsupported {objective_func = }"
        if objective_func == "all":
            for f in TVMTuner.SUPPORTED_OBJECTIVE_FUNCS:
                self.insert_top_n_netlet_to_storage(f, top)
            return
        for i, kernel_source in enumerate(self._get_top_n_sources(objective_func, top)):
            module_function = ModuleKernel(
                self.module_name,
                method=self.method,
                kernel_source=kernel_source,
                platform=self.platform,
                input_infos=self.input_infos,
                output_infos=self.output_infos,
                parameters=self.parameters,
            )
            module_function.dump_to_db(objective_func, i + 1)

    def _get_best_source(self, objective_func: str, rank: int):
        if not self.tune_log_file.exists() and self.old_tune_log_file.exists():
            contents = self.old_tune_log_file.read_text()
            self.tune_log_file.parent.mkdir(parents=True, exist_ok=True)
            self.tune_log_file.write_text(contents)
        record_data = self._get_all_record_data(objective_func, True)
        if rank == 1:
            score, grid_dim, block_dim, inp = min(record_data, key=lambda x: x[0])
        else:
            score, grid_dim, block_dim, inp = type(self)._sort_record_data_by_group(
                record_data
            )[rank - 1]
        kernel_source = self._get_kernel_source_from_measure_input(
            inp, grid_dim, block_dim
        )
        return kernel_source

    def _get_top_n_sources(self, objective_func: str, top: int):
        if not self.tune_log_file.exists() and self.old_tune_log_file.exists():
            contents = self.old_tune_log_file.read_text()
            self.tune_log_file.parent.mkdir(parents=True, exist_ok=True)
            self.tune_log_file.write_text(contents)
        record_data = self._get_all_record_data(objective_func, True)
        top_records = type(self)._sort_record_data_by_group(record_data)
        for i in range(top):
            kernel_source = self._get_kernel_source_from_measure_input(
                top_records[i][3], top_records[i][1], top_records[i][2]
            )
            yield kernel_source

    def _get_all_record_data(self, objective_func: str, enable_mp: bool = True):
        record_reader = auto_scheduler.RecordReader(str(self.tune_log_file))
        record_data = []
        if enable_mp:
            # pathos.multiprocessing
            from pathos import multiprocessing as mp

            pool = mp.ProcessPool()
            async_results = []
            for inp, res in record_reader:
                ar = pool.apipe(
                    _calcu_record_mp,
                    self.tvm_task.compute_dag,
                    objective_func,
                    inp.serialize(),
                    res,
                )
                async_results.append([ar, inp])
            pool.close()
            pool.join()
            for ar, inp in async_results:
                record_data.append((*ar.get(), inp))
            pool.clear()
        else:
            for inp, res in record_reader:
                record_data.append(self._calcu_record(objective_func, inp, res))
        return record_data

    def _calcu_record(self, objective_func, inp, res):
        # for inp, res in record_reader:
        tvm_sch, tvm_args = self.tvm_task.compute_dag.apply_steps_from_state(inp.state)
        tvm_ir = tvm.lower(tvm_sch, tvm_args, simple_mode=True)
        grid_dim, block_dim = parse_culaunch_config(tvm_ir)
        if objective_func == "fastest":
            score = sum(res.costs) / len(res.costs)
        elif objective_func == "most_efficient":
            num_blocks = grid_dim[0] * grid_dim[1] * grid_dim[2]
            num_threads = block_dim[0] * block_dim[1] * block_dim[2]
            score = num_blocks * sum(res.costs) / len(res.costs)
        else:
            raise ValueError(f"Unsupported {objective_func = }")
        return (score, grid_dim, block_dim, inp)

    def _get_kernel_source_from_measure_input(
        self, inp: auto_scheduler.MeasureInput, grid_dim, block_dim
    ):
        source_code = self._get_source_code_from_measure_input(inp)
        culaunch_config = make_culaunch_config_str(grid_dim, block_dim)
        kernel_source = culaunch_config + source_code
        return kernel_source

    def _get_source_code_from_measure_input(self, inp: auto_scheduler.MeasureInput):
        sch, args = self.tvm_task.compute_dag.apply_steps_from_state(
            inp.state, self.tvm_task.layout_rewrite_option
        )
        func = tvm.driver.build_module.build(sch, args, "cuda")
        source_code = func.imported_modules[0].get_source()
        return source_code

    @staticmethod
    def _sort_record_data_by_group(record_data: list):
        record_data.sort(key=lambda x: x[1:3])
        top_records = []
        for _, g in groupby(record_data, key=lambda x: x[1:3]):
            g = list(g)
            g.sort(key=lambda x: x[0])
            top_records.append(g[0])
        top_records.sort(key=lambda x: x[0])
        return top_records


def _calcu_record_mp(
    compute_dag: auto_scheduler.ComputeDAG,
    objective_func: str,
    inp: auto_scheduler.MeasureInput,
    res: auto_scheduler.MeasureResult,
):
    input = auto_scheduler.MeasureInput.deserialize(inp)
    # for inp, res in record_reader:
    tvm_sch, tvm_args = compute_dag.apply_steps_from_state(input.state)
    tvm_ir = tvm.lower(tvm_sch, tvm_args, simple_mode=True)
    grid_dim, block_dim = parse_culaunch_config(tvm_ir)
    costs = list(res.costs)
    if objective_func == "fastest":
        score = sum(costs) / len(costs)
    elif objective_func == "most_efficient":
        num_blocks = grid_dim[0] * grid_dim[1] * grid_dim[2]
        num_threads_per_block = block_dim[0] * block_dim[1] * block_dim[2]
        if num_threads_per_block % 32 != 0:
            score = float("inf")
        else:
            from math import ceil

            # costs[0] *= num_blocks * ceil(num_threads_per_block / 32)
            # costs[0] *= num_blocks * num_threads_per_block
            score = (
                sum(costs) / len(costs) * num_blocks * ceil(num_threads_per_block / 32)
            )
            # score = sum(costs) / len(costs) * num_blocks
    else:
        raise ValueError(f"Unsupported {objective_func = }")
    return (score, grid_dim, block_dim)
