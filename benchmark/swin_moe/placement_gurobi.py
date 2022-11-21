# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from itertools import product

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import argparse


def load_scatter_trace(trace_path):
    return np.load(trace_path, allow_pickle=True)


class PlacementSolver:
    def __init__(
        self, nodes: int, scatter_trace, least_path_per_node: int = 0, mode="optimizer"
    ):
        self.nodes = nodes
        self.scatter_trace = scatter_trace
        self.scatter_num = len(self.scatter_trace) + 1
        self.least_path_per_node = least_path_per_node if least_path_per_node > 0 else 1
        self.path_nums = [
            len(self.scatter_trace[i]) for i in range(self.scatter_num - 1)
        ]
        self.path_nums.append(len(self.scatter_trace[self.scatter_num - 1]))
        self.mode = mode
        self.build_model()

    def build_model(self):
        self.model = gp.Model("placement")
        self.construct_variable()
        self.construct_objective()
        self.add_constraints()

    def solve(self):
        self.model.Params.LogToConsole = True
        self.model.Params.MIPGap = 0.05
        # self.model.Params.TimeLimit = 60
        # self.model.Params.IterationLimit = 1
        self.model.optimize()
        print("Obj: ", self.model.objVal)

    def construct_variable(self):
        self.placements = [
            self.model.addMVar(
                (self.path_nums[i], self.nodes), vtype=GRB.BINARY, name=f"placement_{i}"
            )
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

    def export_results(self):
        results = []
        for i in range(self.scatter_num):
            print(np.argmax(np.array(self.placements[i].x), axis=1))
            results.append(np.argmax(np.array(self.placements[i].x), axis=1))
        results = np.array(results)
        np.save(f"results_{self.nodes}.npy", results)
        # return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=4)
    args = parser.parse_args()
    least_path_per_node = 16 // args.nodes
    scatter_trace = load_scatter_trace("scatter_trace.npy")

    solver = PlacementSolver(
        args.nodes, scatter_trace, least_path_per_node, mode="optimizer"
    )
    solver.solve()
    solver.export_results()


if __name__ == "__main__":
    main()
