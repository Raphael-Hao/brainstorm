# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
from brt._C.router import (
    generate_indices_and_loads,
    dispatch_with_indices_and_loads,
    dispatch_with_dst_indices_1d,
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

route_indices, dst_loads = generate_indices_and_loads(
    hot_mask, supported_capacities, capacity_padding=True, is_dst_index=True
)

in_data = torch.randn((8, path_num)).cuda()
print(in_data)
print(route_indices)
print(dst_loads)

# out_data_1 = dispatch_with_indices_and_loads(in_data, route_indices, dst_loads)
# print(out_data_1)
out_data_2 = dispatch_with_indices_and_loads(
    in_data, route_indices, dst_loads, max_path_padding=True
)
print(out_data_2)
out_data_2 = dispatch_with_dst_indices_1d(in_data, route_indices, dst_loads)
print(out_data_2)
# torch.allclose(out_data_1[0], out_data_2)