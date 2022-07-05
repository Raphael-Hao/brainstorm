# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import brt.frontend.nn as nn
import torch
from brt.common import log
from brt.frontend import symbolize
from brt.routers import GatherRouter, ScatterRouter

# log.set_level("routers", "DEBUG")


class RouterModel(nn.Module):
    def __init__(self, score_func, protocol_type, route_logic, **kwargs):
        super().__init__()
        self.score_func = score_func
        self.scatter_router = ScatterRouter(
            path_num=2, protocol_type=protocol_type, route_logic=route_logic, **kwargs
        )
        self.expert1 = nn.Identity()
        self.expert2 = nn.Identity()
        self.gather_router = GatherRouter(path_num=2, sparse=False)

    def forward(self, x):
        score = self.score_func(x)
        route_results = self.scatter_router(x, score)
        x_0 = self.expert1(route_results[0])
        x_1 = self.expert2(route_results[1])
        x = self.gather_router([x_0, x_1])
        return x_0, x_1, x


class SparseRouterModel(nn.Module):
    def __init__(self, score_func, protocol_type, route_logic, **kwargs):
        super().__init__()
        self.score_func = score_func
        self.scatter_router = ScatterRouter(
            path_num=2, protocol_type=protocol_type, route_logic=route_logic, **kwargs
        )
        self.expert1 = nn.Identity()
        self.expert2 = nn.Identity()
        self.gather_router = GatherRouter(path_num=2, sparse=True)

    def forward(self, x):
        score = self.score_func(x)
        route_results = self.scatter_router(x, score)
        x_0 = self.expert1(route_results[0])
        x_1 = self.expert2(route_results[1])
        x = self.gather_router([x_0, x_1])
        return x_0, x_1, x


class RouterTest(unittest.TestCase):
    def all_to_single(self, Model, inputs, dst):
        def score_func(inputs):
            gates = torch.zeros(
                inputs.shape[0], 2, dtype=torch.int64, device=inputs.device
            )
            gates[:, dst] = 1
            return gates

        model = Model(
            score_func=score_func, protocol_type="threshold", route_logic="1d"
        )

        results = model(inputs)
        self.assertTrue(torch.allclose(results[dst].data, inputs))
        self.assertTrue(results[1 - dst].data.numel() == 0)
        self.assertTrue(torch.allclose(results[2].data, inputs))

    def drop_half_single(self, Model, inputs, dst, which_half, sparse=False):
        def score_func(inputs):
            gates = torch.zeros(
                inputs.shape[0], 2, dtype=torch.int64, device=inputs.device
            )
            if which_half == "upper":
                gates[inputs.shape[0] // 2 :, dst] = 1
            else:
                gates[: inputs.shape[0] // 2, dst] = 1
            return gates

        model = Model(
            score_func=score_func, protocol_type="threshold", route_logic="1d"
        )
        results = model(inputs)
        self.assertTrue(results[1 - dst].data.numel() == 0)
        if which_half == "upper":
            self.assertTrue(
                torch.allclose(results[dst].data, inputs[inputs.size(0) // 2 :])
            )
            if sparse:
                self.assertTrue(
                    torch.allclose(results[2].data, inputs[inputs.size(0) // 2 :])
                )
            else:
                self.assertTrue(
                    torch.allclose(
                        results[2].data[inputs.size(0) // 2 :],
                        inputs[inputs.size(0) // 2 :],
                    )
                )
        else:
            self.assertTrue(torch.allclose(results[dst], inputs[: inputs.size(0) // 2]))
            if sparse:
                self.assertTrue(
                    torch.allclose(results[2].data, inputs[: inputs.size(0) // 2])
                )
            else:
                self.assertTrue(
                    torch.allclose(
                        results[2].data[: inputs.size(0) // 2],
                        inputs[: inputs.size(0) // 2],
                    )
                )

    def test_2d_route(self):
        for i in range(2):
            x = torch.arange(0, 40, dtype=torch.float32).view(4, 10)
            self.all_to_single(RouterModel, x, i)
            self.drop_half_single(RouterModel, x, i, "upper")
            self.drop_half_single(RouterModel, x, i, "lower")
            self.all_to_single(SparseRouterModel, x, i)
            self.drop_half_single(SparseRouterModel, x, i, "upper", True)
            self.drop_half_single(SparseRouterModel, x, i, "lower", True)

    def test_3d_route(self):
        for i in range(2):
            x = torch.arange(0, 40, dtype=torch.float32).view(2, 2, 10)
            self.all_to_single(RouterModel, x, i)
            self.drop_half_single(RouterModel, x, i, "upper")
            self.drop_half_single(RouterModel, x, i, "lower")
            self.all_to_single(SparseRouterModel, x, i)
            self.drop_half_single(SparseRouterModel, x, i, "upper", True)
            self.drop_half_single(SparseRouterModel, x, i, "lower", True)

    def test_4d_route(self):
        for i in range(2):
            x = torch.arange(0, 40, dtype=torch.float32).view(4, 1, 10, 1)
            self.all_to_single(RouterModel, x, i)
            self.drop_half_single(RouterModel, x, i, "upper")
            self.drop_half_single(RouterModel, x, i, "lower")
            self.all_to_single(SparseRouterModel, x, i)
            self.drop_half_single(SparseRouterModel, x, i, "upper", True)
            self.drop_half_single(SparseRouterModel, x, i, "lower", True)

    def test_script(self):
        score_func = nn.Sequential(nn.Linear(10, 2), nn.Softmax(dim=1))
        protocol_type = "topk"

        def jit_script(Model):
            model = Model(
                score_func=score_func, protocol_type=protocol_type, route_logic="1d"
            )
            try:
                script_simple_net = torch.jit.script(symbolize(model))
                print(script_simple_net.graph)
                print(script_simple_net.inlined_graph)
            except Exception as e:
                self.fail(f"Failed to inline the jitted graph: {e}")

        jit_script(RouterModel)
        jit_script(SparseRouterModel)


if __name__ == "__main__":
    unittest.main()
