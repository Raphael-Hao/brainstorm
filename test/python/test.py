# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

#%%
import torch

device_name = torch.cuda.get_device_name(0)
print(device_name)


def get_device_name(device_type, device_id=0):
    if device_type == "CPU":
        return "Default_CPU"
    elif device_type == "CUDA_GPU":
        raw_name = torch.cuda.get_device_name(device_id)
        return raw_name.replace(" ", "_").replace("-", "_")
    else:
        raise ValueError("Unknown device type: {}".format(device_type))


print(get_device_name("CUDA_GPU"))

# %%
op_type = "Conv2d"
method = "forward"
device_name = "Default_CPU"
fname = "_".join([op_type, method, device_name])
print(fname)

# %%
