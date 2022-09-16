# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import List
import torch
from brt.runtime.benchmark import CUDATimer, CPUTimer


def async_copy(
    datas: List[torch.Tensor], stream: torch.cuda.Stream, event: torch.cuda.Event
):
    with torch.cuda.stream(stream):
        for i, data in enumerate(datas):
            datas[i] = data.cuda(non_blocking=True)
        event.record(stream)
    torch.cuda.current_stream().wait_event(event)


timer = CUDATimer(loop=100, repeat=5)


for i in range(0, 11):
    data_num = int(1024 / (2**i))
    datas = [torch.randn(2**i, 1024, 1024) for _ in range(data_num)]
    copy_stream = torch.cuda.Stream()
    copy_event = torch.cuda.Event()
    timer.execute(lambda: async_copy(datas, copy_stream, copy_event), msg=f"{data_num}x{2**i}x1024")
