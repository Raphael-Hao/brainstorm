# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import time
from typing import Iterator, List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from brt.runtime import BRT_CACHE_PATH, BRT_LOG_PATH

__all__ = [
    "profile",
    "BenchmarkArgumentManager",
    "Benchmarker",
    "Timer",
    "CUDATimer",
    "CPUTimer",
]


def profile_v2(model: nn.Module, data_collection, vendor: str):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=5),
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            (BRT_LOG_PATH / f"./profile_results/{vendor}").as_posix()
        ),
        record_shapes=True,
    ) as p:
        np.random.seed(0)
        data_indices = np.random.randint(0, len(data_collection), 10)

        with torch.inference_mode():
            for step in range(10):
                model(**data_collection[data_indices[step]])
                p.step()


def profile(func):
    profiler = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=1, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            str(BRT_LOG_PATH / "model_logs/brt_dr")
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )
    torch.cuda.synchronize()
    with torch.inference_mode():
        profiler.start()
        for _i in range(12):
            func()
            torch.cuda.synchronize()
            profiler.step()
        profiler.stop()


class ResultWriter:
    def __init__(self, fname, clean=False):
        self.root = True
        if dist.is_initialized():
            self.root = dist.get_rank() == 0
        if not fname.endswith(".csv"):
            fname += ".csv"
        self.fpath = BRT_CACHE_PATH / "results" / fname
        self.fpath.parent.mkdir(parents=True, exist_ok=True)
        if clean:
            self.clean()

    def clean(self):
        if self.root:
            if self.fpath.exists():
                self.fpath.unlink()

    def write(self, result: str):
        if self.root:
            binary_io = self.fpath.open("a")
            binary_io.write(result + "\n")
            binary_io.close()


class Timer:
    def __init__(
        self,
        warm_up: int = 5,
        loop=10,
        repeat=1,
        detail=False,
        export_fname: str = None,
        clean_export=False,
    ) -> None:
        self.root = True
        if dist.is_initialized():
            self.root = dist.get_rank() == 0
        self.set_configure(warm_up, loop, repeat, detail, export_fname, clean_export)

    def set_configure(
        self,
        warm_up: int = 5,
        loop=10,
        repeat=1,
        detail=False,
        export_fname: str = None,
        clean_export=False,
    ):
        self.warm_up = warm_up
        self.loop = loop
        self.repeat = repeat
        self.detail = detail
        self.result_writer = ResultWriter(export_fname, clean_export)

    def start(self):
        self.elapsed = []

    def stop(self):
        self.avg = np.mean(self.elapsed)
        self.max = np.max(self.elapsed)
        self.min = np.min(self.elapsed)

    def step_start(self):
        raise NotImplementedError

    def step_stop(self):
        self.elapsed.append(self.step_elapsed)

    def print(self, msg):
        if self.root:
            if self.detail == False:
                print(
                    f"{msg} test results in ms: avg: {self.avg:.3f} max: {self.max:.3f} min: {self.min:.3f}"
                )
            else:
                print(
                    f"Configuration: warm_up: {self.warm_up} loop: {self.loop} repeat: {self.repeat}",
                    f"{msg} test results in ms: avg: {self.avg:.3f} max: {self.max:.3f} min: {self.min:.3f}",
                )

    def export(self, msg):
        if self.root:
            self.result_writer.write(
                f"{msg},{self.avg:.3f},{self.max:.3f},{self.min:.3f}"
            )

    def execute(self, func, msg, export=False):
        with torch.inference_mode():
            for _i in range(self.warm_up):
                func()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.start()
            for _i in range(self.repeat):
                self.step_start()
                for _ in range(self.loop):
                    func()
                self.step_stop()
            self.stop()
            self.print(msg)
            if export:
                self.export(msg)

    def memory_execute(self, model, x, msg, export=False):
        assert self.loop == 1
        with torch.inference_mode():
            for _i in range(self.warm_up):
                model(x)
            model.cpu()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.start()
            for _i in range(self.repeat):
                self.step_start()
                for _ in range(self.loop):
                    model(x)
                self.step_stop()
                model.cpu()
            self.stop()
            self.print(msg)
            if export:
                self.export(msg)


class CPUTimer(Timer):
    def __init__(
        self,
        warm_up: int = 5,
        loop=10,
        repeat=1,
        detail=False,
        export_fname: str = None,
        clean_export=False,
    ) -> None:
        super().__init__(warm_up, loop, repeat, detail, export_fname, clean_export)
        self.start_time = None
        self.end_time = None

    def step_start(self):
        self.start_time = time.time()

    def step_stop(self):
        self.step_elapsed = (time.time() - self.start_time) / self.loop
        super().step_stop()


class CUDATimer(Timer):
    def __init__(
        self,
        warm_up: int = 5,
        loop=10,
        repeat=1,
        detail=False,
        export_fname: str = None,
        clean_export=False,
    ) -> None:
        super().__init__(warm_up, loop, repeat, detail, export_fname, clean_export)
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def step_start(self):
        torch.cuda.synchronize()
        self.start_event.record(torch.cuda.current_stream())

    def step_stop(self):
        self.end_event.record(torch.cuda.current_stream())
        self.end_event.synchronize()
        self.step_elapsed = self.start_event.elapsed_time(self.end_event) / self.loop
        super().step_stop()


class MemoryStats:
    @staticmethod
    def reset_cuda_stats():
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def print_cuda_stats():
        print(torch.cuda.memory_summary())


class BenchmarkArgumentManager:
    def __init__(self, argparser: argparse.ArgumentParser = None) -> None:
        if argparser is None:
            argparser = argparse.ArgumentParser()
        self.argparser = argparser
        self.benchmark = self.argparser.add_argument(
            "--benchmark", choices=[], nargs="+", help="benchmark item"
        )

    def add_item(self, item_name) -> argparse.ArgumentParser:
        self.benchmark.choices.append(item_name)

    def get_parser(self) -> argparse.ArgumentParser:
        return self.argparser

    def get_args(self) -> argparse.Namespace:
        return self.argparser.parse_args()


class Benchmarker:
    def __init__(self) -> None:
        self.benchmark_num = 0

    def add_benchmark(self, name, func):
        self.benchmark_num += 1
        setattr(self, name, func)

    def benchmarking(self, benchmark_items):
        if benchmark_items is None:
            return
        for item in benchmark_items:
            getattr(self, item)()


def deterministic_random_generator(
    shape, num=1, dtype=None, device=None, seed=None
) -> Iterator[torch.Tensor]:
    if seed is None:
        if dist.is_initialized():
            seed = dist.get_rank()
        else:
            seed = 0
    np.random.seed(seed)
    i = 0
    while i < num:
        i += 1
        yield torch.from_numpy(np.random.randn(*shape)).type(dtype).to(device)
