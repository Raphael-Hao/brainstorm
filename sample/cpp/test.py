# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

#%%
import torch
from brt._C.router import (
    generate_indices_and_loads,
    dispatch_with_indices_and_loads,
)

path_num = 4
gates = torch.randn((8, path_num)).cuda()
topk_indices = torch.topk(gates, k=1, dim=1).indices

hot_mask = (
    torch.zeros(gates.size(0), path_num, dtype=torch.int32, device=gates.device,)
    .scatter_(1, topk_indices, 1)
    .cuda()
)

supported_capacities = torch.Tensor([1, 2, 4, 8, 16]).int().cuda()
# supported_capacities = None

seat_indices, loads = generate_indices_and_loads(
    hot_mask, is_tag_index=False
)
print(seat_indices, loads)

in_data = torch.randn((8, path_num)).cuda()
print(in_data)
print(seat_indices)
print(loads)

# out_data_1 = dispatch_with_indices_and_loads(in_data, route_indices, dst_loads)
# print(out_data_1)
out_data_2 = dispatch_with_indices_and_loads(
    in_data, seat_indices, loads, max_path_padding=True
)
print(out_data_2)

# torch.allclose(out_data_1[0], out_data_2)
# %%
