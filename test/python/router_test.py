# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import brt.nn as nn
import torch
from brt.primitive.helper import symbolize
from brt.router import (
    GatherRouter,
    ScatterRouter,
    SparseGatherRouter,
    SparseScatterRouter,
    TagRouter,
)


class RouterModel(nn.Module):
    def __init__(self, route_func, route_method):
        super().__init__()
        self.scatter_router = ScatterRouter(
            route_num=2, route_func=route_func, route_method=route_method
        )
        self.expert1 = nn.Identity()
        self.expert2 = nn.Identity()
        self.gather_router = GatherRouter(route_num=2)

    def forward(self, x):
        route_results, route_tags, loads = self.scatter_router(x)
        x_0 = self.expert1(route_results[0])
        x_1 = self.expert2(route_results[1])
        x = self.gather_router([x_0, x_1], route_tags, loads)
        return x_0, x_1, x


class SparseRouterModel(nn.Module):
    def __init__(self, route_func, route_method):
        super().__init__()
        self.tag_router = TagRouter()
        self.scatter_router = SparseScatterRouter(
            route_num=2, route_func=route_func, route_method=route_method
        )
        self.expert1 = nn.Identity()
        self.expert2 = nn.Identity()
        self.gather_router = SparseGatherRouter(route_num=2)

    def forward(self, x):
        x, tags = self.tag_router(x)
        route_results, route_tags = self.scatter_router(x, tags)
        x_0 = self.expert1(route_results[0])
        x_1 = self.expert2(route_results[1])
        x, tags = self.gather_router([x_0, x_1], route_tags)
        return x_0, x_1, x


class RouterTest(unittest.TestCase):
    def all_to_single_route(self, Model, inputs, dst):
        def route_func(inputs):
            gates = torch.zeros(
                inputs.shape[0], 2, dtype=torch.int64, device=inputs.device
            )
            gates[:, dst] = 1
            return gates

        model = Model(
            route_func=route_func,
            route_method="threshold",
        )
        results = model(inputs)
        self.assertTrue(torch.allclose(results[dst], inputs))
        self.assertTrue(results[1 - dst].numel() == 0)
        self.assertTrue(torch.allclose(results[2], inputs))

    def drop_half_samples(self, Model, inputs, dst, which_half, sparse=False):
        def route_func(inputs):
            gates = torch.zeros(
                inputs.shape[0], 2, dtype=torch.int64, device=inputs.device
            )
            if which_half == "upper":
                gates[inputs.shape[0] // 2 :, dst] = 1
            else:
                gates[: inputs.shape[0] // 2, dst] = 1
            return gates

        model = Model(
            route_func=route_func,
            route_method="threshold",
        )
        results = model(inputs)
        self.assertTrue(results[1 - dst].numel() == 0)
        if which_half == "upper":
            self.assertTrue(torch.allclose(results[dst], inputs[inputs.size(0) // 2 :]))
            if sparse:
                self.assertTrue(
                    torch.allclose(results[2], inputs[inputs.size(0) // 2 :])
                )
            else:
                self.assertTrue(
                    torch.allclose(
                        results[2][inputs.size(0) // 2 :], inputs[inputs.size(0) // 2 :]
                    )
                )
        else:
            self.assertTrue(torch.allclose(results[dst], inputs[: inputs.size(0) // 2]))
            if sparse:
                self.assertTrue(
                    torch.allclose(results[2], inputs[: inputs.size(0) // 2])
                )
            else:
                self.assertTrue(
                    torch.allclose(
                        results[2][: inputs.size(0) // 2], inputs[: inputs.size(0) // 2]
                    )
                )

    def test_2d_route(self):
        for i in range(2):
            x = torch.arange(0, 40, dtype=torch.float32).view(4, 10)
            self.all_to_single_route(RouterModel, x, i)
            self.drop_half_samples(RouterModel, x, i, "upper")
            self.drop_half_samples(RouterModel, x, i, "lower")
            self.all_to_single_route(SparseRouterModel, x, i)
            self.drop_half_samples(SparseRouterModel, x, i, "upper", True)
            self.drop_half_samples(SparseRouterModel, x, i, "lower", True)

    # def test_3dforward(self):
    #     x = torch.arange(0, 30, dtype=torch.float32).view(3, 10)
    #     for _ in range(10):
    #         x = model(x)
    #     return x

    # def test_4dforward(self):
    #     model = Router2D()
    #     x = torch.arange(0, 30, dtype=torch.float32).view(3, 10)
    #     for _ in range(10):
    #         x = model(x)
    #     return x

    def test_script(self):
        route_func = nn.Sequential(nn.Linear(10, 2), nn.Softmax(dim=1))
        route_method = "topk"

        def jit_script(Model):
            model = Model(route_func, route_method)
            try:
                script_simple_net = torch.jit.script(symbolize(model))
                script_simple_net.inlined_graph
                print(script_simple_net.inlined_graph)
            except Exception as e:
                self.fail(f"Failed to inline the jitted graph: {e}")

        jit_script(RouterModel)
        jit_script(SparseRouterModel)

    def test_traced(self):
        pass


if __name__ == "__main__":
    unittest.main()
