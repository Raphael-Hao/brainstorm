# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from itertools import product

import torch.nn as nn
import numpy as np
import gurobipy as gp
from gurobipy import GRB

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
        self.model = gp.Model("placement")
        self.construct_variable()
        self.construct_objective()
        self.add_constraints()

    def solve(self):
        self.model.Params.LogToConsole = True
        self.model.Params.MIPGap =0.001
        # self.model.Params.TimeLimit = 60
        self.model.optimize()
        print("Obj: ",self.model.objVal)

    def construct_variable(self):
        self.placements = [
            self.model.addMVar((self.path_nums[i], self.nodes), vtype=GRB.BINARY, name=f"placement_{i}")
            for i in range(self.scatter_num)
        ]

    def construct_objective(self):
        cost = None
        for i in range(self.scatter_num - 1):
            for path_i, path_j in product(
                range(self.path_nums[i]), range(self.path_nums[i + 1])
            ):
                # if_same_node = np.zeros(1, dtype=np.int32)
                is_same_node = 0
                for node in range(self.nodes):
                    is_same_node += (
                        self.placements[i][path_i, node]
                        * self.placements[i + 1][path_j, node]
                    )
                path_cost = (1 - is_same_node) * self.scatter_trace[i][path_i][path_j]
                if cost is None:
                    cost = path_cost
                else:
                    cost = cost + path_cost
        self.model.setObjective(cost, GRB.MINIMIZE)

    def add_constraints(self):
        for i in range(self.scatter_num):
            for j in range(self.path_nums[i]):
                self.model.addConstr(self.placements[i][j, :].sum() == 1)
            for j in range(self.nodes):
                self.model.addConstr(
                    self.placements[i][:, j].sum() >= self.least_path_per_node
                )


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
