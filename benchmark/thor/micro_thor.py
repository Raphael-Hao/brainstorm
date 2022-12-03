# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

# %%
import torch

from thor_config import ThorConfig
from thor_moe import ThorMoE, FusedThorMoE
from brt.runtime.benchmark import BenchmarkArgumentManager, CUDATimer


def main():
    bench_arg_manager = BenchmarkArgumentManager()
    parser = bench_arg_manager.get_parser()
    parser.add_argument("--bench", type=str, default="brt", choices=["brt", "brt_homo"])
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
    config.expert_type = args.bench

    if args.bench == "brt":
        thor_moe = ThorMoE(config).eval()
    elif args.bench == "brt_homo":
        thor_moe = FusedThorMoE(config).eval()

    thor_moe.cuda()

    x = torch.randn(1, 4, config.hidden_size).cuda()
    x = thor_moe(x)
    print(x[0].shape)
    x = torch.randn(1, args.token, config.hidden_size).cuda()
    x = thor_moe(x)
    print(x[0].shape)

    cuda_timer = CUDATimer(loop=100, repeat=10)

    # %%

    x = torch.randn(1, args.token, 512).cuda()
    cuda_timer.execute(
        lambda: thor_moe(x),
        msg=f"{args.bench},{args.expert},{args.token}",
        export=True,
        export_path=f"thor/micro_results.csv",
    )


if __name__ == "__main__":
    main()
