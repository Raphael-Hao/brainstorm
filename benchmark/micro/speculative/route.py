# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import time

import torch
import torch.nn as nn
from brt import Annotator
from brt.app.rand import MissHitScatter
from brt.router import GatherRouter
from brt.runtime import BRT_CACHE_PATH
from brt.runtime.benchmark import CUDATimer


class DefaultModel(nn.Module):
    def __init__(self, path_num=2, in_features=1, router_time=0.1):
        super().__init__()
        self.in_features = in_features
        self.path_num = path_num
        self.router_time = router_time
        self.selected_path = path_num - 1
        self.router_time
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
        time.sleep(self.router_time)
        for i, branch in enumerate(self.branches):
            if i == self.selected_path:
                x = branch(x)
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
    def __init__(self, path_num=2, in_features=1, is_hit=False, router_time=0.1):
        super().__init__()
        self.unroll_index = int(router_time // 0.00087) * 4
        # self.unroll_index = 4
        self.in_features = in_features
        self.path_num = path_num
        self.router_time = router_time
        if is_hit:
            self.selected_path = 0
        else:
            self.selected_path = path_num - 1
        # self.scatter = MissHitScatter(path_num=self.path_num, is_hit=is_hit)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    *[nn.Linear(self.in_features, self.in_features) for _ in range(20)]
                )
                for _path_id in range(self.path_num)
            ]
        )
        self.former_half_branches = self.branches[0][0 : self.unroll_index]
        self.latter_half_branches = self.branches[0][self.unroll_index :]
        self.half_index = 4 + self.unroll_index

    def forward(self, x):
        half_x = self.former_half_branches(x)
        # time.sleep(self.router_time)
        if self.selected_path != 0:
            time.sleep(self.router_time)
            for i in range(1, self.path_num):
                if i == self.selected_path:
                    x = self.branches[i](x)
                    break
        else:
            x = self.latter_half_branches(half_x)
        # x =self.former_half_branches(x)
        return half_x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-size", type=int, default=100)
    parser.add_argument("--path-num", type=int, default=2)
    parser.add_argument("--time", type=float, default=0.1)
    args = parser.parse_args()
    default_model = (
        DefaultModel(args.path_num, args.cell_size, router_time=args.time).eval().cuda()
    )
    # hit_model = HitModel(args.path_num, args.cell_size).eval().cuda()
    hit_model = (
        MissModel(args.path_num, args.cell_size, is_hit=True, router_time=args.time)
        .eval()
        .cuda()
    )
    miss_model = (
        MissModel(args.path_num, args.cell_size, is_hit=False, router_time=args.time)
        .eval()
        .cuda()
    )
    annotator = Annotator(dims=[0])

    x = torch.randn(1, args.cell_size).cuda()
    x = annotator(x)
    result_path = BRT_CACHE_PATH / "results" / "micro" / "speculative" / "route_e2e.csv"

    timer = CUDATimer(repeat=10, loop=100, export_fname=result_path.as_posix())

    def default_forward():
        default_model(x)

    def hit_forward():
        hit_model(x)

    def miss_forward():
        miss_model(x)

    timer.execute(
        default_forward,
        f"Default,{args.time*1e3}",
        export=True,
    )
    timer.execute(
        hit_forward,
        f"Hit,{args.time*1e3}",
        export=True,
    )
    timer.execute(
        miss_forward,
        f"Miss,{args.time*1e3}",
        export=True,
    )


if __name__ == "__main__":
    main()
