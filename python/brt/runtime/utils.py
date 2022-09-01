# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import time
import torch


class Timer:
    def __init__(self) -> None:
        pass

    def start():
        raise NotImplementedError

    def stop():
        raise NotImplementedError

    def print(self, msg, iterations=1):
        raise NotImplementedError


class CPUTimer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def print(self, msg, iterations=1):
        print(f"{msg}: {(self.end_time - self.start_time) * 1000/iterations:.2f} ms")


class GPUTimer:
    def __init__(self) -> None:
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        self.start_event.record(torch.cuda.current_stream())

    def stop(self):
        self.end_event.record(torch.cuda.current_stream())
        self.end_event.synchronize()

    def print(self, msg, iterations=1):
        print(
            f"{msg}: {self.start_event.elapsed_time(self.end_event) / iterations:.2f} ms"
        )


class BenchmarkContext:
    def __init__(self) -> None:
        pass


class Benchmarker:
    def __init__(self) -> None:
        pass

    def benchmarking(self):
        pass
