# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import argparse

import brt.runtime.distributed as brt_dist
import torch
import torch.distributed as dist
from brt.runtime import BRT_CACHE_PATH
from brt.runtime.benchmark import CUDATimer


def main():
    dist.init_process_group(backend="nccl")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark", type=str, default="brt", choices=["imbalance", "balance"]
    )
    parser.add_argument(
        "--local-expert",
        type=int,
        default=1,
        help="group size",
        choices=[1, 2, 4, 8, 16, 32, 64, 128],
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=4,
        help="grain size",
        choices=[32, 64, 128, 256, 512, 1024, 2048],
    )
    parser.add_argument(
        "--load",
        type=int,
        default=1024,
        help="grain size",
        choices=[32, 64, 128, 256, 512, 1024, 2048],
    )
    args = parser.parse_args()

    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    cell_size = args.cell_size
    local_expert = args.local_expert

    cuda_timer = CUDATimer(repeat=10, loop=100, root=local_rank)

    result_path = (
        BRT_CACHE_PATH / "results" / "micro" / "distributed" / "placement_a2a.csv"
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)

    if args.benchmark == "imbalance":
        loads = torch.zeros(
            (world_size * local_expert,), dtype=torch.int32, device=device
        ).fill_(1)
        dst_rank = (local_rank + 1) % world_size
        loads[dst_rank * local_expert : (dst_rank + 1) * local_expert] = args.load
    else:
        loads = torch.zeros(
            (world_size * local_expert,), dtype=torch.int32, device=device
        ).fill_(1)
        loads[local_rank * local_expert : (local_rank + 1) * local_expert] = args.load

    tensor = torch.randn(loads.sum().item(), cell_size, device=device)
    torch.cuda.synchronize()
    dist.barrier()

    def brt_sparse_a2a(tensor, loads):
        out_data, out_loads, in_loads = brt_dist.group_sparse_a2a(tensor, loads)
        final_data = brt_dist.size_known_group_sparse_a2a(out_data, out_loads, in_loads)
        return final_data

    cuda_timer.execute(
        lambda: brt_sparse_a2a(tensor, loads),
        msg=f"{args.benchmark},{world_size},{local_expert},{cell_size},{args.load}",
        export=True,
        export_path=result_path,
    )


if __name__ == "__main__":
    main()
