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
        "--benchmark", type=str, default="brt", choices=["brt", "torch"]
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

    result_path = (
        BRT_CACHE_PATH / "results" / "micro" / "distributed" / "sparse_e2e.csv"
    )
    cuda_timer = CUDATimer(repeat=4, loop=100, export_fname=result_path.as_posix())

    if args.benchmark == "brt":
        torch.random.manual_seed(local_rank)

        loads = torch.randint(
            args.load, (world_size * local_expert,), dtype=torch.int32, device=device
        )

        tensor = torch.randn(loads.sum().item(), cell_size, device=device)
        torch.cuda.synchronize()
        dist.barrier()

        def brt_sparse_a2a(tensor, loads):
            out_data, out_loads, in_loads = brt_dist.group_sparse_a2a(tensor, loads)
            final_data = brt_dist.size_known_group_sparse_a2a(
                out_data, out_loads, in_loads
            )
            return final_data

        cuda_timer.execute(
            lambda: brt_sparse_a2a(tensor, loads),
            msg=f"BRT,{world_size},{local_expert},{cell_size}",
            export=True,
        )
    else:
        tensor = torch.randn(
            args.load * world_size * local_expert, cell_size, device=device
        )
        torch.cuda.synchronize()
        dist.barrier()

        def torch_a2a(tensor):
            out_data = torch.zeros_like(tensor)
            dist.all_to_all_single(out_data, tensor)
            out_data = out_data.view(world_size, -1, cell_size)
            out_data = out_data.permute(1, 0, 2).contiguous()
            out_data = out_data.permute(0, 2, 1).contiguous()
            out_data = out_data.view(-1, cell_size)
            final_data = torch.empty_like(out_data)
            dist.all_to_all_single(final_data, out_data)
            return final_data

        cuda_timer.execute(
            lambda: torch_a2a(tensor),
            msg=f"Torch,{world_size},{local_expert},{cell_size}",
            export=True,
        )


if __name__ == "__main__":
    main()
