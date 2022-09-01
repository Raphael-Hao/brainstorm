# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import time

import torch
import torch.nn as nn
from brt.runtime.weight_load import WeightLoader


class SimpleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10000, 10000)

    def forward(self, x):
        x = self.linear(x)
        return x


start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
new_cuda_stream = torch.cuda.Stream()


simple_net = SimpleNet()
in_data = torch.randn(1, 10000)
origin_out_data = simple_net(in_data)
# init the weight loader
WeightLoader.init()

pinned_simple_net = WeightLoader.pin_memory(simple_net)
WeightLoader.inject_placement_check(simple_net)
pinned_out_data = pinned_simple_net(in_data)

# first load and unload
torch.cuda.synchronize()
cuda_simple_net = WeightLoader.load_module(pinned_simple_net)

with torch.cuda.stream(new_cuda_stream):
    for i in range(100):
        cuda_out_data = cuda_simple_net(in_data.cuda(non_blocking=True))


unload_simple_net = WeightLoader.unload_module(cuda_simple_net)
unload_out_data = unload_simple_net(in_data)
# second load and unload
torch.cuda.synchronize()

start_time = time.time()
cuda_simple_net = WeightLoader.load_module(unload_simple_net)
end_time = time.time()
print(f"cpu time of loading pinned model: {(end_time - start_time) * 1000} ms")


start_time = time.time()
with torch.cuda.stream(new_cuda_stream):
    cuda_out_data = cuda_simple_net(in_data.cuda(non_blocking=True))
end_time = time.time()
print(f"cpu time of forwarding a loaded model: {(end_time - start_time) * 1000} ms")

unload_simple_net = WeightLoader.unload_module(cuda_simple_net)
unload_out_data = unload_simple_net(in_data)


start_time = time.time()
unload_simple_net.cuda()
end_time = time.time()
print(f"cpu time of loading model through .cuda(): {(end_time - start_time) * 1000} ms")
