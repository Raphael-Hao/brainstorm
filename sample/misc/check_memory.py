import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.conv = nn.Conv2d(3, 3, 3)

    def forward(self, x):
        x = self.linear(x)
        x = self.conv(x)
        return x

simple_net =SimpleNet()
simple_net.eval()
in_data = torch.randn(1,3,10,10)
with torch.inference_mode():
    origin_out_data = simple_net(in_data)

from brt.runtime.mem_plan import load_module, pin_memory, unload_module

print(f"Allocated: {torch.cuda.memory_allocated()}")
print(f"Reserved: {torch.cuda.memory_reserved()}")

print(f"Allocated: {torch.cuda.memory_allocated()}")
print(f"Reserved: {torch.cuda.memory_reserved()}")

pinned_simple_net = pin_memory(simple_net)
pinned_simple_net.eval()
with torch.inference_mode():
    pinned_out_data = pinned_simple_net(in_data)

print(f"Allocated: {torch.cuda.memory_allocated()}")
print(f"Reserved: {torch.cuda.memory_reserved()}")

cuda_simple_net = load_module(pinned_simple_net)
print(f"Allocated: {torch.cuda.memory_allocated()}")
print(f"Reserved: {torch.cuda.memory_reserved()}")


cuda_simple_net.eval()
cuda_in_data = in_data.cuda()
print(f"Allocated: {torch.cuda.memory_allocated()}")
print(f"Reserved: {torch.cuda.memory_reserved()}")

with torch.inference_mode():
    cuda_out_data = cuda_simple_net(cuda_in_data)
print(f"Allocated: {torch.cuda.memory_allocated()}")
print(f"Reserved: {torch.cuda.memory_reserved()}")
print(torch.cuda.memory_summary())
cuda_in_data=None
# cuda_out_data = None

print(f"Allocated: {torch.cuda.memory_allocated()}")
print(f"Reserved: {torch.cuda.memory_reserved()}")

unload_simple_net = unload_module(cuda_simple_net)
print(f"Allocated: {torch.cuda.memory_allocated()}")
print(f"Reserved: {torch.cuda.memory_reserved()}")

unload_simple_net.eval()
with torch.inference_mode():
    unload_out_data = unload_simple_net(in_data)

# torch.cuda.empty_cache()
print(f"Allocated: {torch.cuda.memory_allocated()}")
print(f"Reserved: {torch.cuda.memory_reserved()}")

print(torch.cuda.memory_summary())