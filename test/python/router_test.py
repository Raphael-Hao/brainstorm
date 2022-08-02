# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import torch
import torch.nn as nn

# from brt.runtime import log
from brt.router import GatherRouter, ScatterRouter
from brt.trace.graph import GraphTracer

# log.set_level("router", "DEBUG")


class BranchRoute(nn.Module):
    def __init__(
        self,
        gate,
        dispatch_score=False,
        sparse=False,
    ):
        super().__init__()
        self.gate = gate
        self.scatter_router = ScatterRouter(
            dispatch_score=dispatch_score,
            protocol_type="threshold",
            protocol_kwargs={"threshold": 0.5},
        )
        self.expert1 = nn.Identity()
        self.expert2 = nn.Identity()
        self.gather_router = GatherRouter(
            fabric_type="combine", fabric_kwargs={"sparse": sparse}
        )

    def forward(self, x):
        score = self.gate(x)
        routed_results = self.scatter_router(x, score)
        x_0 = self.expert1(routed_results[0])
        x_1 = self.expert2(routed_results[1])
        x = self.gather_router([x_0, x_1])
        return x_0, x_1, x


class WeightedBranchRoute(BranchRoute):
    def __init__(
        self,
        gate,
        sparse=False,
    ):
        super().__init__(gate, dispatch_score=True, sparse=sparse)

    def forward(self, x):
        score = self.gate(x)
        routed_results, routed_score = self.scatter_router(x, score)
        x_0 = self.expert1(routed_results[0])
        x_1 = self.expert2(routed_results[1])

        weighted_x_0 = x_0 * routed_score[0]
        weighted_x_1 = x_1 * routed_score[1]
        pre_x = self.gather_router([weighted_x_0, weighted_x_1])

        x = self.gather_router([x_0, x_1])
        score = self.gather_router([routed_score[0], routed_score[1]])
        post_x = x * score

        return pre_x, post_x


class RouterTest(unittest.TestCase):
    # def all_to_single(self, Model: BranchRoute, inputs, dst, sparse=False):
    #     def gate(inputs):
    #         gates = torch.zeros(
    #             inputs.shape[0], 2, dtype=torch.float, device=inputs.device
    #         )
    #         gates[:, dst] = 1
    #         return gates

    #     model = Model(gate=gate, sparse=sparse)

    #     results = model(inputs)
    #     self.assertTrue(torch.allclose(results[dst].data, inputs))
    #     self.assertTrue(results[1 - dst].data.numel() == 0)
    #     self.assertTrue(torch.allclose(results[2].data, inputs))

    def simple_route(
        self,
        Model: BranchRoute,
        inputs: torch.Tensor,
        dst: int,
        which_half: str = None,
        sparse=False,
    ):
        def gate(inputs):
            gates = torch.zeros(
                inputs.shape[0], 2, dtype=torch.float, device=inputs.device
            )
            if which_half == "upper":
                gates[inputs.shape[0] // 2 :, dst] = 1
            elif which_half == "lower":
                gates[: inputs.shape[0] // 2, dst] = 1
            else:
                gates[:, dst] = 1
            return gates

        model = Model(gate=gate, sparse=sparse)
        results = model(inputs)

        if which_half is None:
            self.assertTrue(torch.allclose(results[dst].data, inputs))
            self.assertTrue(results[1 - dst].data.numel() == 0)
            self.assertTrue(torch.allclose(results[2].data, inputs))
        else:
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
            elif which_half == "lower":
                self.assertTrue(
                    torch.allclose(results[dst], inputs[: inputs.size(0) // 2])
                )
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

    def weighted_route(
        self,
        Model,
        inputs: torch.Tensor,
        dst: int,
        which_half: str = None,
        sparse=False,
    ):
        def gate(inputs):
            gates = torch.zeros(
                inputs.shape[0], 2, dtype=torch.int64, device=inputs.device
            )
            if which_half == "upper":
                gates[inputs.shape[0] // 2 :, dst] = 1
            else:
                gates[: inputs.shape[0] // 2, dst] = 1
            return gates

        model = Model(gate=gate, sparse=sparse)
        results = model(inputs)
        if which_half == "upper":
            self.assertTrue(
                torch.allclose(results[0].data, inputs[inputs.size(0) // 2 :])
            )
            self.assertTrue(
                torch.allclose(results[1].data, inputs[inputs.size(0) // 2 :])
            )
            self.assertTrue(torch.allclose(results[0].data, results[1].data))
        else:
            self.assertTrue(torch.allclose(results[0], inputs[: inputs.size(0) // 2]))
            self.assertTrue(torch.allclose(results[1], inputs[: inputs.size(0) // 2]))
            self.assertTrue(torch.allclose(results[0].data, results[1].data))

    def test_2d_route(self):
        for i in range(2):
            x = torch.arange(0, 80, dtype=torch.float32).view(8, 10)
            self.simple_route(BranchRoute, x, dst=i, sparse=False)
            self.simple_route(BranchRoute, x, dst=i, which_half="upper", sparse=False)
            self.simple_route(BranchRoute, x, dst=i, which_half="lower", sparse=False)
            self.simple_route(BranchRoute, x, dst=i, sparse=True)
            self.simple_route(BranchRoute, x, dst=i, which_half="upper", sparse=True)
            self.simple_route(BranchRoute, x, dst=i, which_half="lower", sparse=True)

    def test_3d_route(self):
        for i in range(2):
            x = torch.arange(0, 160, dtype=torch.float32).view(8, 2, 10)
            self.simple_route(BranchRoute, x, dst=i, sparse=False)
            self.simple_route(BranchRoute, x, dst=i, which_half="upper", sparse=False)
            self.simple_route(BranchRoute, x, dst=i, which_half="lower", sparse=False)
            self.simple_route(BranchRoute, x, dst=i, sparse=True)
            self.simple_route(BranchRoute, x, dst=i, which_half="upper", sparse=True)
            self.simple_route(BranchRoute, x, dst=i, which_half="lower", sparse=True)

    def test_4d_route(self):
        for i in range(2):
            x = torch.arange(0, 320, dtype=torch.float32).view(8, 2, 10, 2)
            self.simple_route(BranchRoute, x, dst=i, sparse=False)
            self.simple_route(BranchRoute, x, dst=i, which_half="upper", sparse=False)
            self.simple_route(BranchRoute, x, dst=i, which_half="lower", sparse=False)
            self.simple_route(BranchRoute, x, dst=i, sparse=True)
            self.simple_route(BranchRoute, x, dst=i, which_half="upper", sparse=True)
            self.simple_route(BranchRoute, x, dst=i, which_half="lower", sparse=True)

    def test_trace(self):
        gate = nn.Sequential(nn.Linear(10, 2), nn.Softmax(dim=1))

        def jit_script(Model):
            model = Model(gate=gate)
            try:
                graph_module = GraphTracer().trace(model)
            #     script_simple_net = torch.jit.script(symbolize(model))
            #     print(script_simple_net.graph)
            #     print(script_simple_net.inlined_graph)
            except Exception as e:
                self.fail(f"Failed to trace the graph through torch.fx: {e}")

        jit_script(BranchRoute)
        jit_script(WeightedBranchRoute)


if __name__ == "__main__":
    unittest.main()
