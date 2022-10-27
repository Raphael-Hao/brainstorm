# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from itertools import product

import torch.nn as nn
import numpy as np
import cvxpy as cp

from brt.passes.base import PassBase, register_pass
from brt.router import is_router


def dump_scatter_trace(mod: nn.Module):
    scatter_results = []
    for m_name, m in mod.named_modules():
        if is_router(m) and "scatter" in m._router_type:
            scatter_results.append(np.array(m.ptu_decision_history, dtype=object))
    return scatter_results
    # np.save("scatter_results.npy", scatter_results, allow_pickle=True)


def load_scatter_trace(trace_path):
    return np.load(trace_path, allow_pickle=True)


class PlacementSolver:
    def __init__(
        self, nodes: int, scatter_trace, least_path_per_node: int = 0, mode="optimizer"
    ):
        self.nodes = nodes
        self.scatter_trace = scatter_trace
        self.scatter_num = len(self.scatter_trace)
        self.least_path_per_node = least_path_per_node if least_path_per_node > 0 else 1
        self.path_nums = [len(self.scatter_trace[i]) for i in range(self.scatter_num)]
        self.mode = mode
        self.build_model()

    def build_model(self):
        self.construct_variable()
        self.construct_objective()
        self.add_constraints()

    def solve(self):
        self.problem = cp.Problem(self.objective, self.constraints)
        self.problem.solve(solver=cp.OSQP, verbose=True)
        # self.problem.solve(solver=cp.GUROBI, verbose=True)
        # print(self.problem.status)

    def construct_variable(self):
        self.placements = [
            cp.Variable((self.path_nums[i], self.nodes), boolean=True)
            for i in range(self.scatter_num)
        ]

    def construct_objective(self):
        cost = None
        for i in range(self.scatter_num - 1):
            for path_i, path_j in product(
                range(self.path_nums[i]), range(self.path_nums[i + 1])
            ):
                path_cost = (
                    1
                    - cp.sum(
                        cp.multiply(
                            self.placements[i][path_i],
                            self.placements[i + 1][path_j],
                        )
                    )
                ) * self.scatter_trace[i][path_i][path_j]
                if cost is None:
                    cost = path_cost
                else:
                    cost = cost + path_cost
        self.objective = cp.Minimize(cost)

    def add_constraints(self):
        constrains = []
        for i in self.placements:
            constrains.append(cp.sum(i, axis=1) == 1)
            constrains.append(cp.sum(i, axis=0) >= self.least_path_per_node)
        self.constraints = constrains


def main():
    scatter_trace = load_scatter_trace("scatter_trace.npy")
    solver = PlacementSolver(4, scatter_trace, least_path_per_node=4, mode="optimizer")
    solver.construct_variable()
    solver.construct_objective()
    solver.add_constraints()
    solver.solve()


@register_pass("pipline")
class PipelinePass(PassBase):
    pass


@register_pass("sharded")
class ShardedPass(PassBase):
    pass


if __name__ == "__main__":
    main()
