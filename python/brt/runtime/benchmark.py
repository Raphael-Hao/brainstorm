# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import time
import torch
import argparse

__all__ = ["BenchmarkArgumentManager", "Benchmarker"]


class Timer:
    def __init__(self, itrations: int = 0) -> None:
        self.iterations = itrations

    def set_iterations(self, iterations: int):
        self.iterations = iterations

    def start():
        raise NotImplementedError

    def stop():
        raise NotImplementedError

    def print(self, msg):
        raise NotImplementedError

    def execute(self, func, msg):
        self.start()
        for _ in range(self.iterations):
            func()
        self.stop()
        self.print(msg)


class CPUTimer(Timer):
    def __init__(self, iterations: int = 0) -> None:
        super().__init__(iterations)
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def print(self, msg):
        print(
            f"{msg}: {(self.end_time - self.start_time) * 1000/self.iterations:.2f} ms"
        )


class CUDATimer(Timer):
    def __init__(self, iterations: int = 0) -> None:
        super().__init__(iterations)
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        self.start_event.record(torch.cuda.current_stream())

    def stop(self):
        self.end_event.record(torch.cuda.current_stream())
        self.end_event.synchronize()

    def print(self, msg):
        print(
            f"{msg}: {self.start_event.elapsed_time(self.end_event) / self.iterations:.2f} ms"
        )


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
