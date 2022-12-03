# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import math

# %%
import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from brt.app.rand import RandScatter, UniformScatter  # pylint: disable=unused-import
from brt.jit import make_jit_kernel
from brt.router import GatherRouter
from brt.runtime.benchmark import BenchmarkArgumentManager, CUDATimer
from thor_config import ThorConfig


class FusedThorExpert(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.token_num = config.token_num
        max_exp = math.ceil(math.log2(self.token_num // config.expert_num))
        # capacities = [2**i for i in range(max_exp)]
        capacities = [2**max_exp]
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        dense1s = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.intermediate_size)
                for _ in range(config.expert_num)
            ]
        )
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        dense2s = nn.ModuleList(
            [
                nn.Linear(config.intermediate_size, config.hidden_size)
                for _ in range(config.expert_num)
            ]
        )
        sample_inputs = [torch.ones(i, config.hidden_size) for i in capacities]

        self.expert1_kernel = make_jit_kernel(
            dense1s, sample_inputs, opt_level="homo_fuse"
        )

        sample_inputs = [torch.ones(i, config.intermediate_size) for i in capacities]

        self.expert2_kernel = make_jit_kernel(
            dense2s, sample_inputs, opt_level="homo_fuse"
        )

        self.expert1_standalone_inputs = []
        for linear in dense1s:
            self.expert1_standalone_inputs.extend([linear.weight, linear.bias])
        self.expert1_standalone_inputs = nn.ParameterList(
            self.expert1_standalone_inputs
        )
        self.expert2_standalone_inputs = []
        for linear in dense2s:
            self.expert2_standalone_inputs.extend([linear.weight, linear.bias])
        self.expert2_standalone_inputs = nn.ParameterList(
            self.expert2_standalone_inputs
        )

    def forward(
        self,
        inter_state: torch.Tensor,
    ) -> torch.Tensor:
        capacities = inter_state.loads.tolist()
        route_indices = inter_state.route_indices
        score = inter_state.score
        expert1_out = torch.empty(
            inter_state.shape[0], self.intermediate_size, device=inter_state.device
        )
        self.expert1_kernel(
            shared_inputs=[inter_state, expert1_out],
            standalone_inputs=self.expert1_standalone_inputs,
            capacities=capacities,
        )
        # x = expert1_out
        expert2_out = torch.empty(
            inter_state.shape[0], self.hidden_size, device=inter_state.device
        )
        self.expert2_kernel(
            shared_inputs=[expert1_out, expert2_out],
            standalone_inputs=self.expert2_standalone_inputs,
            capacities=capacities,
        )
        expert2_out.route_indices = route_indices
        expert2_out.loads = inter_state.loads
        expert2_out.score = score
        return expert2_out


class FusedThorMoE(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.token_num = config.token_num
        self.expert_num = config.expert_num
        max_exp = math.ceil(math.log2(self.token_num))
        capacities = [2**i for i in range(max_exp)]
        self.scatter_router = UniformScatter(
            path_num=config.expert_num,
            fabric_type="homo_fused_dispatch",
            protocol_kwargs={
                "supported_capacities": torch.tensor(capacities, dtype=torch.int32),
            },
        )
        self.gather_router = GatherRouter(
            fabric_type="homo_fused_combine",
        )
        self.fused_expert = FusedThorExpert(config)

    def forward(self, hidden_states):
        inter_states = hidden_states.view(-1, hidden_states.size(-1))
        x = self.scatter_router(inter_states)
        x = self.fused_expert(x)

        def expert_forward():
            self.fused_expert(x)

        cuda_timer = CUDATimer(loop=100, repeat=10)
        cuda_timer.execute(
            lambda: expert_forward(),
            msg=f"brt_homo,{self.expert_num},{self.token_num}",
            export=True,
            export_path=f"thor/micro_results.csv",
        )
        inter_states = self.gather_router(x)
        inter_states = inter_states.view(
            hidden_states.size(0), hidden_states.size(1), hidden_states.size(2)
        )
        return inter_states


class ThorExpert(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, inter_state):
        x = self.dense1(inter_state)
        x = self.dense2(x)
        return x


class ThorMoE(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.expert_num = config.expert_num
        self.token_num = config.token_num
        self.experts = nn.ModuleList(
            [ThorExpert(config) for _ in range(config.expert_num)]
        )
        self.rand_scatter = UniformScatter(path_num=config.expert_num)
        self.gather_router = GatherRouter()

    def forward(self, hidden_states):
        # B x T x H -> T x H
        inter_states = hidden_states.view(-1, hidden_states.size(-1))
        route_results = self.rand_scatter(inter_states)
        expert_results = []
        for i, expert in enumerate(self.experts):
            expert_results.append(expert(route_results[i]))

        def expert_forward():
            expert_results = []
            for i, expert in enumerate(self.experts):
                expert_results.append(expert(route_results[i]))

        cuda_timer = CUDATimer(loop=100, repeat=10)
        cuda_timer.execute(
            lambda: expert_forward(),
            msg=f"brt,{self.expert_num},{self.token_num}",
            export=True,
            export_path=f"thor/micro_results.csv",
        )
        inter_states = self.gather_router(expert_results)
        inter_states = inter_states.view(
            hidden_states.size(0), hidden_states.size(1), hidden_states.size(2)
        )
        x = inter_states
        return x


def main():
    bench_arg_manager = BenchmarkArgumentManager()
    parser = bench_arg_manager.get_parser()
    parser.add_argument("--bench", type=str, default="brt", choices=["brt", "brt_homo"])
    parser.add_argument("--expert", type=int, default=2)
    parser.add_argument("--token", type=int, default=32)
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

    x = torch.randn(1, args.token, config.hidden_size).cuda()
    x = thor_moe(x)
    print(x[0].shape)


if __name__ == "__main__":
    main()
