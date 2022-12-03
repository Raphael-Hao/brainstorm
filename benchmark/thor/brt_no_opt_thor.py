# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

# %%
import torch

from thor_config import ThorConfig
from thor_model import ThorEncoder
from brt.runtime.benchmark import BenchmarkArgumentManager, CUDATimer


def main():
    bench_arg_manager = BenchmarkArgumentManager()
    parser = bench_arg_manager.get_parser()
    parser.add_argument("--expert", type=int, default=2, choices=[2, 4, 8, 16])
    parser.add_argument("--token", type=int, default=32, choices=[32, 64, 96, 128, 160, 192, 224, 256])
    args = bench_arg_manager.get_args()
    config = ThorConfig()
    config.token_num = args.token
    config.hidden_size = 512
    config.intermediate_size = 1024
    config.num_attention_heads = 8
    config.num_hidden_layers = 12
    config.expert_num = args.expert
    config.expert_type = "brt_moe"

    # thor_moe = ThorMoE(config).eval()
    thor_moe = ThorEncoder(config).eval()

    thor_moe.cuda()

    cuda_timer = CUDATimer(loop=100, repeat=10)

    # %%
    x = torch.zeros(1, args.token, 512).cuda()
    cuda_timer.execute(
        lambda: thor_moe(x),
        msg=f"brt,{args.expert},{args.token}",
        export=True,
        export_path=f"thor/all_results.csv",
    )


if __name__ == "__main__":
    main()
