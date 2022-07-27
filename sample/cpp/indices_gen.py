from brt._C.router import generate_dst_indices, generate_src_indices
from tutel.jit_kernels.gating import fast_cumsum_sub_one
import torch

dst_num = 4
gates = torch.randn((4, 4)).cuda()
topk_indices = torch.topk(gates, k=1, dim=1).indices

hot_mask = (
    torch.zeros(
        gates.size(0),
        dst_num,
        dtype=torch.int64,
        device=gates.device,
    )
    .scatter_(1, topk_indices, 1)
    .cuda()
)

supported_capacities = torch.Tensor([1, 2, 4, 8, 16]).int().cuda()
indices = fast_cumsum_sub_one(hot_mask, dim=0)
local_indices, dst_loads = generate_dst_indices(hot_mask.to(torch.int32))
print(local_indices)
print(dst_loads)
