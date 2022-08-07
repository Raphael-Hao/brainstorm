# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from brt.router.utils import generate_dst_indices
import torch

dst_num = 4
gates = torch.randn((4, 4)).cuda()
topk_indices = torch.topk(gates, k=1, dim=1).indices

route_indices = (
    torch.zeros(
        gates.size(0),
        dst_num,
        dtype=torch.int64,
        device=gates.device,
    )
    .scatter_(1, topk_indices, 1)
    .cuda()
)

supported_capacities = torch.Tensor([0, 1, 2, 4, 8, 16]).int().cuda()

local_indices, dst_loads = generate_dst_indices(route_indices, supported_capacities)
print(local_indices)
print(dst_loads)

from brt._C.router import dispatch_with_dst_indices_1d
import torch

in_data = torch.randn(4, 4, 4, dtype=torch.float32, requires_grad=True).cuda()

total_load = torch.sum(dst_loads).item()

new_gates = torch.ones_like(gates, dtype=torch.float32).cuda()

print(in_data)
print(local_indices)
print(dst_loads)
out_data = dispatch_with_dst_indices_1d(
    in_data,
    local_indices,
    dst_loads,
    True,
)

print(in_data)
print(out_data)
