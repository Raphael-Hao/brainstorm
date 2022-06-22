# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

#%%
import torch
from brt.common import BRT_KERNEL_TEMPLATE_PATH, log
from brt.jit import CUDACompiler, make_jit_kernel

log.set_level("jit", "DEBUG")

l_1 = torch.nn.Linear(512, 1024)
l_2 = torch.nn.Linear(512, 1024)
l_3 = torch.nn.Linear(512, 1024)
l_4 = torch.nn.Linear(512, 1024)

modules = torch.nn.ModuleList([l_1, l_2, l_3, l_4]).cuda()

sample_inputs = []
sample_input_1 = torch.ones(1, 512).cuda()
sample_inputs.append(sample_input_1)
sample_input_2 = torch.ones(2, 512).cuda()
sample_inputs.append(sample_input_2)
sample_input_4 = torch.ones(4, 512).cuda()
sample_inputs.append(sample_input_4)

sample_output_1 = torch.zeros(1, 1024).cuda()
sample_output_2 = torch.zeros(2, 1024).cuda()
sample_output_4 = torch.zeros(4, 1024).cuda()

# homo_kernel_source = (
#     BRT_KERNEL_TEMPLATE_PATH
#     / "processed_LinearBias__input_0_1_512__output_0_1_1024__in_features_512_out_features_1024_3_4_homo_fuse_human.cu"
# )
# homo_jit_kernel = CUDACompiler.create_raw(homo_kernel_source.read_text())

homo_jit_kernel = make_jit_kernel(
    modules, sample_inputs, "forward", opt_level="homo_fuse"
)
#%%


weight_0 = torch.ones((1024, 512), device="cuda")
bias_0 = torch.zeros((1024,), device="cuda")
weight_1 = torch.ones((1024, 512), device="cuda")
bias_1 = torch.zeros((1024,), device="cuda")
weight_2 = torch.ones((1024, 512), device="cuda")
bias_2 = torch.zeros((1024,), device="cuda")
weight_3 = torch.ones((1024, 512), device="cuda")
bias_3 = torch.zeros((1024,), device="cuda")

print(f"weight_0 data ptr: {hex(weight_0.data_ptr())}")
print(f"bias_0 data ptr: {hex(bias_0.data_ptr())}")
print(f"weight_1 data ptr: {hex(weight_1.data_ptr())}")
print(f"bias_1 data ptr: {hex(bias_1.data_ptr())}")
print(f"weight_2 data ptr: {hex(weight_2.data_ptr())}")
print(f"bias_2 data ptr: {hex(bias_2.data_ptr())}")
print(f"weight_3 data ptr: {hex(weight_3.data_ptr())}")
print(f"bias_3 data ptr: {hex(bias_3.data_ptr())}")

torch.cuda.synchronize()
stream = torch.cuda.current_stream()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

#%%

jit_kernel_1 = make_jit_kernel(l_1, sample_inputs=sample_input_1)
jit_kernel_1(sample_input_1, weight_0, sample_output_1, bias_0)
torch.cuda.synchronize()

start_event.record(stream)
for i in range(100):
    jit_kernel_1(sample_input_1, weight_0, sample_output_1, bias_0)
end_event.record(stream)
stream.synchronize()
print("capacity 1: {:.3f}".format(start_event.elapsed_time(end_event) / 100))

print(sample_output_1)

#%%
jit_kernel_2 = make_jit_kernel(l_1, sample_inputs=sample_input_2)
jit_kernel_2(sample_input_2, weight_0, sample_output_2, bias_0)
torch.cuda.synchronize()

start_event.record(stream)
for i in range(1000):
    jit_kernel_2(sample_input_2, weight_0, sample_output_2, bias_0)
end_event.record(stream)
stream.synchronize()
print("capacity 2: {:.3f}".format(start_event.elapsed_time(end_event) / 1000))
print(sample_output_2)

#%%

jit_kernel_4 = make_jit_kernel(l_1, sample_inputs=sample_input_4)
jit_kernel_4(sample_input_4, weight_0, sample_output_4, bias_0)
torch.cuda.synchronize()

start_event.record(stream)
for i in range(1000):
    jit_kernel_4(sample_input_4, weight_0, sample_output_4, bias_0)
end_event.record(stream)
stream.synchronize()
print("capacity 4: {:.3f}".format(start_event.elapsed_time(end_event) / 1000))
print(sample_output_4)


#%%
torch.set_printoptions(edgeitems=16)

data_0 = torch.ones((16, 512), device="cuda")
outdata_0 = torch.zeros((16, 1024), device="cuda")
print(f"data_0 data ptr: {hex(data_0.data_ptr())}")
print(f"outdata_0 data ptr: {hex(outdata_0.data_ptr())}")

outdata_0.zero_()

shared_inputs = [data_0, outdata_0]
standalone_inputs = [
    weight_0,
    bias_0,
    weight_1,
    bias_1,
    weight_2,
    bias_2,
    weight_3,
    bias_3,
]
start_event.record(stream)
branch_capacities = [2, 2, 2, 2]
homo_jit_kernel(
    shared_inputs=shared_inputs,
    standalone_inputs=standalone_inputs,
    capacities=branch_capacities,
)
print(data_0)
print(outdata_0)
end_event.record(stream)
stream.synchronize()
print("first time: {:.3f}".format(start_event.elapsed_time(end_event)))

#%%
test_loop = 1000
start_event.record(stream)
for i in range(test_loop):
    homo_jit_kernel(
        shared_inputs=shared_inputs,
        standalone_inputs=standalone_inputs,
        capacities=branch_capacities,
    )
end_event.record(stream)
stream.synchronize()
print(outdata_0)
print("forward time: {:.3f}".format(start_event.elapsed_time(end_event) / test_loop))

# %%
