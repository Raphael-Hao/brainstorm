# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import math

# %%
import torch
import torch.nn as nn
from brt.jit import make_jit_kernel
from brt.runtime.benchmark import BenchmarkArgumentManager, CUDATimer
from thor_config import ThorConfig

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


class FusedThorExpert(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        capacities = [config.token_num]
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        dense1s = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.intermediate_size)
                for _ in range(config.expert_num)
            ]
        )
        dense2s = nn.ModuleList(
            [
                nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
                for _ in range(config.expert_num)
            ]
        )
        sample_inputs = [torch.ones(i, config.hidden_size) for i in capacities]

        self.expert1_kernel = make_jit_kernel(
            dense1s, sample_inputs, opt_level="homo_fuse", rank=10
        )

        sample_inputs = [torch.ones(i, config.intermediate_size) for i in capacities]

        # self.expert2_kernel = make_jit_kernel(
        #     dense2s, sample_inputs, opt_level="homo_fuse"
        # )

        self.expert1_standalone_inputs = []
        for linear in dense1s:
            self.expert1_standalone_inputs.extend([linear.weight, linear.bias])
        self.expert1_standalone_inputs = nn.ParameterList(
            self.expert1_standalone_inputs
        )
        self.expert2_standalone_inputs = []
        for linear in dense2s:
            self.expert2_standalone_inputs.extend([linear.weight])
        self.expert2_standalone_inputs = nn.ParameterList(
            self.expert2_standalone_inputs
        )

    def forward(
        self,
        inter_state: torch.Tensor,
        capacities,
    ) -> torch.Tensor:
        expert1_out = torch.empty(
            inter_state.shape[0], self.intermediate_size, device=inter_state.device
        )
        self.expert1_kernel(
            shared_inputs=[inter_state, expert1_out],
            standalone_inputs=self.expert1_standalone_inputs,
            capacities=capacities,
        )
        # # x = expert1_out
        # expert2_out = torch.empty(
        #     inter_state.shape[0], self.hidden_size, device=inter_state.device
        # )
        # self.expert2_kernel(
        #     shared_inputs=[expert1_out, expert2_out],
        #     standalone_inputs=self.expert2_standalone_inputs,
        #     capacities=capacities,
        # )
        return expert1_out


class ThorExpert(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, inter_state):
        x = self.dense1(inter_state)
        # x = self.dense2(x)
        return x


class ThorMoE(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.expert_num = config.expert_num
        self.experts = nn.ModuleList(
            [ThorExpert(config) for _ in range(config.expert_num)]
        )

    def forward(self, route_results):
        # B x T x H -> T x H
        expert_results = []
        for i, expert in enumerate(self.experts):
            expert_results.append(expert(route_results[i]))

        expert_results = torch.cat(expert_results, dim=0).contiguous()
        # print(expert_results.shape)
        return expert_results


class BatchedMatmul(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        dense1s = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.intermediate_size)
                for _ in range(config.expert_num)
            ]
        )
        dense2s = nn.ModuleList(
            [
                nn.Linear(config.intermediate_size, config.hidden_size)
                for _ in range(config.expert_num)
            ]
        )
        linear1_weights, linear1_bias = [], []
        for linear in dense1s:
            linear1_weights.append(linear.weight)
            linear1_bias.append(linear.bias)
        linear2_weights, linear2_bias = [], []
        for linear in dense2s:
            linear2_weights.append(linear.weight)
            linear2_bias.append(linear.bias)
        linear1_weight = torch.cat(linear1_weights, dim=0).view(
            config.expert_num, config.hidden_size, config.intermediate_size
        )
        linear1_bias = torch.cat(linear1_bias, dim=0).view(
            config.expert_num, 1, config.intermediate_size
        )
        linear2_weight = torch.cat(linear2_weights, dim=0).view(
            config.expert_num, config.intermediate_size, config.hidden_size
        )
        linear2_bias = torch.cat(linear2_bias, dim=0).view(config.expert_num, 1, -1)
        self.register_parameter("linear1_weight", nn.Parameter(linear1_weight))
        self.register_parameter("linear1_bias", nn.Parameter(linear1_bias))
        self.register_parameter("linear2_weight", nn.Parameter(linear2_weight))
        self.register_parameter("linear2_bias", nn.Parameter(linear2_bias))

    def forward(self, inter_state):
        hidden_state = torch.bmm(inter_state, self.linear1_weight) + self.linear1_bias
        x = torch.bmm(hidden_state, self.linear2_weight) + self.linear2_bias
        return x


def main():
    bench_arg_manager = BenchmarkArgumentManager()
    parser = bench_arg_manager.get_parser()
    parser.add_argument(
        "--bench", type=str, default="brt", choices=["brt", "brt_homo", "matmul"]
    )
    parser.add_argument("--expert", type=int, default=2)
    parser.add_argument("--token", type=int, default=32)
    args = bench_arg_manager.get_args()
    config = ThorConfig()
    config.token_num = args.token  # 1024
    config.hidden_size = 512
    config.intermediate_size = 1024
    config.num_attention_heads = 8
    config.num_hidden_layers = 1
    config.expert_num = args.expert
    config.expert_type = args.bench
    cuda_timer = CUDATimer(loop=100, repeat=10)
    if args.bench == "brt":
        thor_moe = ThorMoE(config).eval().cuda()
        x = [
            torch.randn(config.token_num, config.hidden_size).cuda()
            for _ in range(config.expert_num)
        ]
        x[-1] = torch.randn(config.token_num, config.hidden_size).cuda()
        cuda_timer.execute(
            lambda: thor_moe(x),
            msg=f"brt,{config.expert_num},{config.token_num}",
            export=True,
            export_path=f"thor/micro_results.csv",
        )
    elif args.bench == "brt_homo":
        thor_moe = FusedThorExpert(config).eval().cuda()
        x = torch.randn(
            config.token_num * config.expert_num + config.token_num - config.token_num,
            config.hidden_size,
        ).cuda()
        # x = torch.randn(256 * config.expert_num, 512).cuda()
        capacities = [config.token_num] * (config.expert_num - 1) + [config.token_num]
        cuda_timer.execute(
            lambda: thor_moe(x, capacities),
            msg=f"brt_homo,{config.expert_num},{config.token_num}",
            export=True,
            export_path=f"thor/micro_results.csv",
        )
    elif args.bench == "matmul":
        thor_moe = BatchedMatmul(config).cuda().eval()
        x = torch.randn(config.expert_num, config.token_num, config.hidden_size).cuda()
        thor_moe(x)
        cuda_timer.execute(
            lambda: thor_moe(x),
            msg=f"batched_matmul,{config.expert_num},{config.token_num}",
            export=True,
            export_path=f"thor/micro_results.csv",
        )


if __name__ == "__main__":
    main()
