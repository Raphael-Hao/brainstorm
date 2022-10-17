# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import time
import torch
import argparse
import numpy as np
from brt.runtime import BRT_LOG_PATH

__all__ = [
    "profile",
    "BenchmarkArgumentManager",
    "Benchmarker",
    "Timer",
    "CUDATimer",
    "CPUTimer",
]


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
        for i in range(12):
            func()
            torch.cuda.synchronize()
            profiler.step()
        profiler.stop()


class Timer:
    def __init__(self, warm_up: int = 5, loop=10, repeat=1) -> None:
        self.set_configure(warm_up, loop, repeat)

    def set_configure(self, warm_up: int = 5, loop=10, repeat=1):
        self.warm_up = warm_up
        self.loop = loop
        self.repeat = repeat

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

    def print(self, msg, detail=False):
        if detail == False:
            print(
                f"{msg} test results in ms: avg: {self.avg:.3f} max: {self.max:.3f} min: {self.min:.3f}"
            )
        else:
            print(
                f"{msg} test results in ms: avg: {self.avg:.3f} max: {self.max:.3f} min: {self.min:.3f}",
                f"Configuration: warm_up: {self.warm_up} loop: {self.loop} repeat: {self.repeat}",
            )

    def execute(self, func, msg):
        with torch.inference_mode():
            for i in range(self.warm_up):
                func()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.start()
            for i in range(self.repeat):
                self.step_start()
                for _ in range(self.loop):
                    func()
                self.step_stop()
            self.stop()
            self.print(msg)


class CPUTimer(Timer):
    def __init__(self, warm_up: int = 5, loop=10, repeat=1) -> None:
        super().__init__(warm_up, loop, repeat)
        self.start_time = None
        self.end_time = None

    def step_start(self):
        self.start_time = time.time()

    def step_stop(self):
        self.step_elapsed = (time.time() - self.start_time) / self.loop
        super().step_stop()


class CUDATimer(Timer):
    def __init__(self, warm_up: int = 5, loop=10, repeat=1) -> None:
        super().__init__(warm_up, loop, repeat)
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
