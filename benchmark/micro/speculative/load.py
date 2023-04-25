# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import argparse

import torch
import torch.nn as nn
from brt import Annotator
from brt.app.rand import MissHitScatter
from brt.router import GatherRouter
from brt.runtime import BRT_CACHE_PATH
from brt.runtime.benchmark import CUDATimer


class DefaultModel(nn.Module):
    def __init__(self, path_num=2, in_features=1):
        super().__init__()
        self.in_features = in_features
        self.path_num = path_num
        self.scatter = MissHitScatter(path_num=self.path_num, is_hit=True)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        nn.Linear(
                            in_features=self.in_features,
                            out_features=self.in_features,
                        )
                        for _ in range(20)
                    ]
                )
                for _path_id in range(self.path_num)
            ]
        )
        self.gather = GatherRouter()

    def forward(self, x):
        routed_x = self.scatter(x)
        branch_results = []
        for i, branch in enumerate(self.branches):
            if routed_x[i].numel() > 0:
                branch.to(x.device)
                branch_results.append(branch(routed_x[i]))
        x = self.gather(branch_results)
        return x


class HitModel(nn.Module):
    def __init__(self, path_num=2, in_features=1):
        super().__init__()
        self.in_features = in_features
        self.path_num = path_num
        self.scatter = MissHitScatter(path_num=self.path_num, is_hit=True)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        nn.Linear(
                            in_features=self.in_features,
                            out_features=self.in_features,
                        )
                        for _ in range(20)
                    ]
                )
                for _path_id in range(self.path_num)
            ]
        )
        self.gather = GatherRouter()

    def forward(self, x):
        x = self.branches[0](x)
        return x


class MissModel(nn.Module):
    def __init__(self, path_num=2, in_features=1, is_hit=False):
        super().__init__()
        self.in_features = in_features
        self.path_num = path_num
        self.scatter = MissHitScatter(path_num=self.path_num, is_hit=is_hit)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        nn.Linear(
                            in_features=self.in_features,
                            out_features=self.in_features,
                        )
                        for _ in range(20)
                    ]
                )
                for _path_id in range(self.path_num)
            ]
        )
        self.gather = GatherRouter()

    def forward(self, x):
        routed_x = self.scatter(x)
        branch_results = []
        if routed_x[0].size(0) == 0:
            for i in range(1, len(self.branches)):
                if routed_x[i].numel() > 0:
                    self.branches[i].to(x.device)
                    branch_results.append(self.branches[i](routed_x[i]))
            x = self.gather(branch_results)
        else:
            x = self.branches[0](routed_x[0])
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-size", type=int, default=100)
    parser.add_argument("--path-num", type=int, default=2)
    args = parser.parse_args()
    default_model = DefaultModel(args.path_num, args.cell_size).eval().cpu()
    # hit_model = HitModel(args.path_num, args.cell_size).eval().cuda()
    hit_model = MissModel(args.path_num, args.cell_size, is_hit=True).eval().cpu()
    miss_model = MissModel(args.path_num, args.cell_size, is_hit=False).eval().cpu()

    result_path = BRT_CACHE_PATH / "results" / "micro" / "speculative" / "load_e2e.csv"

    x = torch.randn(1, args.cell_size).cuda()
    annotator = Annotator(dims=[0])
    x = annotator(x)
    timer = CUDATimer(
        warm_up=10, repeat=10, loop=1, export_fname=result_path.as_posix()
    )

    pytorch_total_params = sum(
        p.numel() for p in default_model.branches[0].parameters()
    )
    pytorch_total_params_in_MB = pytorch_total_params * 4 / 1024 / 1024
    print(f"Total params: {pytorch_total_params_in_MB} MB")

    timer.memory_execute(
        default_model,
        x,
        f"Default,{pytorch_total_params_in_MB}",
        export=True,
    )
    hit_model.branches[0].cuda()
    timer.execute(
        lambda: hit_model(x),
        f"Hit,{pytorch_total_params_in_MB}",
        export=True,
    )
    # miss_model.branches[0].cuda()
    timer.memory_execute(
        miss_model,
        x,
        f"Miss,{pytorch_total_params_in_MB}",
        export=True,
    )


if __name__ == "__main__":
    main()
