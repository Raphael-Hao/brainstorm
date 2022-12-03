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
                    *[
                        nn.Conv2d(
                            in_channels=self.in_features,
                            out_channels=self.in_features,
                            kernel_size=1,
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
                        nn.Conv2d(
                            in_channels=self.in_features,
                            out_channels=self.in_features,
                            kernel_size=1,
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
                        nn.Conv2d(
                            in_channels=self.in_features,
                            out_channels=self.in_features,
                            kernel_size=1,
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
        if routed_x[0].size(0) == 0:
            branch_results = []
            for i, branch in enumerate(self.branches[1:]):
                if routed_x[i + 1].numel() > 0:
                    branch.to(x.device)
                branch_results.append(branch(routed_x[i + 1]))
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

    x = torch.randn(1, args.cell_size, 1, 1).cuda()
    timer = CUDATimer(repeat=10, loop=100)

    def default_forward():
        default_model(x)

    def hit_forward():
        hit_model(x)

    def miss_forward():
        miss_model(x)

    result_path = (
        BRT_CACHE_PATH / "results" / "micro" / "speculative" / "load" / f"default.csv"
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    timer.execute(
        default_forward,
        f"{args.path_num},{args.cell_size}",
        export=True,
        export_path=result_path,
    )
    result_path = (
        BRT_CACHE_PATH / "results" / "micro" / "speculative" / "load" / f"hit.csv"
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    hit_model.branches[0].cuda()
    timer.execute(
        hit_forward,
        f"{args.path_num},{args.cell_size}",
        export=True,
        export_path=result_path,
    )
    result_path = (
        BRT_CACHE_PATH / "results" / "micro" / "speculative" / "load" / f"miss.csv"
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    # miss_model.branches[0].cuda()
    timer.execute(
        miss_forward,
        f"{args.path_num},{args.cell_size}",
        export=True,
        export_path=result_path,
    )


if __name__ == "__main__":
    main()
