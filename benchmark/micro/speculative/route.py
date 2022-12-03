# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from brt.app.rand import MissHitScatter
from brt.router import GatherRouter
import torch
import torch.nn as nn
from brt.runtime.benchmark import CUDATimer
from brt.runtime import BRT_CACHE_PATH


class DefaultModel(nn.Module):
    def __init__(self, path_num=2, in_features=1):
        super().__init__()
        self.in_features = in_features
        self.path_num = path_num
        self.scatter = MissHitScatter(path_num=self.path_num, is_hit=True)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    *[nn.Linear(self.in_features, self.in_features) for _ in range(20)]
                )
                for _path_id in range(self.path_num)
            ]
        )
        self.gather = GatherRouter()

    def forward(self, x):
        routed_x = self.scatter(x)
        branch_results = []
        for i, branch in enumerate(self.branches):
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
                    *[nn.Linear(self.in_features, self.in_features) for _ in range(20)]
                )
                for _path_id in range(self.path_num)
            ]
        )
        self.gather = GatherRouter()

    def forward(self, x):
        x = self.branches[0](x)
        return x


class MissModel(nn.Module):
    def __init__(self, path_num=2, in_features=1, unroll_indx=4, is_hit=False):
        super().__init__()
        self.in_features = in_features
        self.path_num = path_num
        self.unroll_indx = unroll_indx
        self.scatter = MissHitScatter(path_num=self.path_num, is_hit=is_hit)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    *[nn.Linear(self.in_features, self.in_features) for _ in range(20)]
                )
                for _path_id in range(self.path_num)
            ]
        )
        self.gather = GatherRouter()

    def forward(self, x):
        self.branches[0][: self.unroll_indx](x)
        routed_x = self.scatter(x)
        if routed_x[0].size(0) == 0:
            branch_results = []
            for i, branch in enumerate(self.branches[1:]):
                branch_results.append(branch(routed_x[i + 1]))
            x = self.gather(branch_results)
        else:
            x = self.branches[0][self.unroll_indx :](routed_x[0])
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-size", type=int, default=100)
    parser.add_argument("--path-num", type=int, default=2)
    parser.add_argument("--unroll-index", type=int, default=4)
    args = parser.parse_args()
    default_model = DefaultModel(args.path_num, args.cell_size).eval().cuda()
    # hit_model = HitModel(args.path_num, args.cell_size).eval().cuda()
    hit_model = (
        MissModel(
            args.path_num, args.cell_size, unroll_indx=args.unroll_index, is_hit=True
        )
        .eval()
        .cuda()
    )
    miss_model = (
        MissModel(
            args.path_num, args.cell_size, unroll_indx=args.unroll_index, is_hit=False
        )
        .eval()
        .cuda()
    )

    x = torch.randn(1, args.cell_size).cuda()
    timer = CUDATimer(repeat=10, loop=100)

    def default_forward():
        default_model(x)

    def hit_forward():
        hit_model(x)

    def miss_forward():
        miss_model(x)

    result_path = (
        BRT_CACHE_PATH / "results" / "micro" / "speculative" / "route" / f"default.csv"
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    timer.execute(
        default_forward,
        f"{args.path_num},{args.unroll_index},{args.cell_size}",
        export=True,
        export_path=result_path,
    )
    result_path = (
        BRT_CACHE_PATH / "results" / "micro" / "speculative" / "route" / f"hit.csv"
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    timer.execute(
        hit_forward,
        f"{args.path_num},{args.unroll_index},{args.cell_size}",
        export=True,
        export_path=result_path,
    )
    result_path = (
        BRT_CACHE_PATH / "results" / "micro" / "speculative" / "route" / f"miss.csv"
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    timer.execute(
        miss_forward,
        f"{args.path_num},{args.unroll_index},{args.cell_size}",
        export=True,
        export_path=result_path,
    )


if __name__ == "__main__":
    main()
