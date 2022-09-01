# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
import time
from torch.utils import data

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)


a = torch.randn(64, 3, 1024, 1024, device="cuda")
torch.cuda.synchronize()

start_stamp = time.time()
b = a.to("cpu", non_blocking=True)
end_stamp = time.time()
print(f"cpu time non blocking to('cpu'): {(end_stamp - start_stamp) * 1000:.2f} ms")

torch.cuda.synchronize()
# start_stamp = time.time()
b = torch.empty_like(a, device="cpu", pin_memory=True)
# b = torch.empty_like(a, device="cpu")
start_stamp = time.time()
b.copy_(a, non_blocking=True)
end_stamp = time.time()
print(f"cpu time copy_-based non blocking to('cpu'): {(end_stamp - start_stamp) * 1000:.2f} ms")

torch.cuda.synchronize()
start_stamp = time.time()
b.copy_(a)
end_stamp = time.time()
print(f"cpu time copy_-based blocking to('cpu'): {(end_stamp - start_stamp) * 1000:.2f} ms")



torch.cuda.synchronize()
start_event.record()
b = a.to("cpu", non_blocking=True)
end_event.record()
end_event.synchronize()
print(f"gpu time non blocking to('cpu'): {start_event.elapsed_time(end_event):.2f} ms")


a = torch.randn(64, 3, 1024, 1024, device="cpu", pin_memory=True)
torch.cuda.synchronize()

start_stamp = time.time()
b = a.to("cuda", non_blocking=True)
end_stamp = time.time()
print(f"cpu time non blocking to('cuda'): {(end_stamp - start_stamp) * 1000:.2f} ms")


torch.cuda.synchronize()
start_event.record()
b = a.to("cuda", non_blocking=True)
end_event.record()
end_event.synchronize()
print(f"gpu time non blocking to('cuda'): {start_event.elapsed_time(end_event):.2f} ms")
