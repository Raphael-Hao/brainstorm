# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Union, List, Type


import unittest

import torch
import torch.nn as nn

from brt import Annotator
from brt.router import GatherRouter, ScatterRouter
from brt.trace.graph import GraphTracer


class BranchRoute(nn.Module):
    def __init__(
        self,
        gate,
        dispatch_score=False,
        single_cell_dispatch=False,
        supported_capacities: Union[List[int], int] = None,
        capacity_padding: bool = False,
        path_wise_padding: bool = False,
    ):
        super().__init__()
        if path_wise_padding and not capacity_padding:
            raise ValueError("path_wise_padding requires capacity_padding")
        self.gate = gate
        self.annotator = Annotator([0])
        self.scatter_router = ScatterRouter(
            dispatch_score=dispatch_score,
            protocol_type="threshold",
            fabric_type="single_cell_dispatch" if single_cell_dispatch else "dispatch",
            protocol_kwargs={"threshold": 0.5},
            fabric_kwargs={
                "supported_capacities": supported_capacities,
                "capacity_padding": capacity_padding,
                "path_wise_padding": path_wise_padding,
            },
        )
        self.expert1 = nn.Identity()
        self.expert2 = nn.Identity()
        self.gather_router = GatherRouter(
            fabric_type="single_cell_combine" if single_cell_dispatch else "combine",
            fabric_kwargs={"ever_padded": capacity_padding},
        )

    def forward(self, x):
        x = self.annotator(x)
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
        single_cell_dispatch=False,
    ):
        super().__init__(
            gate,
            dispatch_score=True,
            single_cell_dispatch=single_cell_dispatch,
        )

    def forward(self, x):
        x = self.annotator(x)
        score = self.gate(x)
        score = self.annotator(score)
        x_dim = x.dim()

        routed_results, routed_score = self.scatter_router(x, score)
        x_0 = self.expert1(routed_results[0])
        x_1 = self.expert2(routed_results[1])

        weighted_x_0 = x_0 * (routed_score[0].view(-1, *([1] * (x_dim - 1))))
        weighted_x_1 = x_1 * (routed_score[1].view(-1, *([1] * (x_dim - 1))))
        pre_x = self.gather_router([weighted_x_0, weighted_x_1])

        x = self.gather_router([x_0, x_1])
        score = self.gather_router([routed_score[0], routed_score[1]])
        post_x = x * score.view(-1, *([1] * (x_dim - 1)))

        return pre_x, post_x


class RouterTest(unittest.TestCase):
    def simple_route(
        self,
        Model: Type[BranchRoute],
        inputs: torch.Tensor,
        dst: int,
        which_half: str = None,
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

        model = Model(gate=gate).eval().cuda()
        for _ in range(5):
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
                    self.assertTrue(
                        torch.allclose(results[2].data, inputs[inputs.size(0) // 2 :])
                    )
                elif which_half == "lower":
                    self.assertTrue(
                        torch.allclose(results[dst].data, inputs[: inputs.size(0) // 2])
                    )
                    self.assertTrue(
                        torch.allclose(results[2].data, inputs[: inputs.size(0) // 2])
                    )

    def weighted_simple_route(
        self,
        Model: Type[WeightedBranchRoute],
        inputs: torch.Tensor,
        dst: int,
        which_half: str = None,
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

        model = Model(gate=gate).eval()
        for _ in range(5):

            results = model(inputs)
            self.assertTrue(torch.allclose(results[0].data, results[1].data))

    def test_generic_route(self):
        multi_dims_sample = [
            torch.arange(0, 80, dtype=torch.float32).view(8, 10).cuda(),
            torch.arange(0, 160, dtype=torch.float32).view(8, 2, 10).cuda(),
            torch.arange(0, 320, dtype=torch.float32).view(8, 2, 10, 2).cuda(),
        ]
        for x in multi_dims_sample:
            for i in range(2):
                self.simple_route(BranchRoute, x, dst=i)
                self.simple_route(BranchRoute, x, dst=i, which_half="upper")
                self.simple_route(BranchRoute, x, dst=i, which_half="lower")
                self.weighted_simple_route(WeightedBranchRoute, x, dst=i)
                self.weighted_simple_route(
                    WeightedBranchRoute, x, dst=i, which_half="upper"
                )
                self.weighted_simple_route(
                    WeightedBranchRoute, x, dst=i, which_half="lower"
                )

    def single_ptu_route(
        self,
        Model: Type[BranchRoute],
        inputs: torch.Tensor,
        dst: Union[int, List[int]],
    ):
        def gate(x):
            assert x.shape[0] == 1, "SinglePTURoute only supports bs==1"
            gates = torch.zeros(1, 2, dtype=torch.float, device=inputs.device)
            if isinstance(dst, List):
                for d in dst:
                    gates[:, d] = 1
            else:
                gates[:, dst] = 1
            return gates

        model = Model(gate=gate, single_cell_dispatch=True).eval()
        for i in range(2):
            if i == 1:
                inputs = inputs.cuda()
                model.cuda()
            results = model(inputs)
            if isinstance(dst, List):
                for d in dst:
                    self.assertTrue(torch.allclose(results[d], inputs))
                self.assertTrue(torch.allclose(results[2], inputs * 2))
            else:
                self.assertTrue(torch.allclose(results[dst], inputs))
                self.assertTrue(results[1 - dst].numel() == 0)
                self.assertTrue(torch.allclose(results[2], inputs))

    def weighted_single_ptu_route(
        self,
        Model: Type[WeightedBranchRoute],
        inputs: torch.Tensor,
        dst: Union[int, List[int]],
    ):
        def gate(x):
            assert x.shape[0] == 1, "SinglePTURoute only supports bs==1"
            gates = torch.zeros(1, 2, dtype=torch.float, device=inputs.device)
            if isinstance(dst, List):
                for d in dst:
                    gates[:, d] = 1
            else:
                gates[:, dst] = 1
            return gates

        model = Model(gate=gate, single_cell_dispatch=True).eval().cuda()
        for _ in range(2):
            results = model(inputs)
            self.assertTrue(torch.allclose(results[0], results[1]))

    def test_single_ptu_route(self):
        multi_dims_sample = [
            torch.arange(0, 10, dtype=torch.float32).view(1, 10).cuda(),
            torch.arange(0, 20, dtype=torch.float32).view(1, 2, 10).cuda(),
            torch.arange(0, 40, dtype=torch.float32).view(1, 2, 10, 2).cuda(),
        ]
        for x in multi_dims_sample:
            self.single_ptu_route(BranchRoute, x, dst=0)
            self.single_ptu_route(BranchRoute, x, dst=1)
            self.single_ptu_route(BranchRoute, x, dst=[0, 1])
            self.weighted_single_ptu_route(WeightedBranchRoute, x, dst=0)
            self.weighted_single_ptu_route(WeightedBranchRoute, x, dst=1)

    def padded_route(
        self,
        Model: Type[BranchRoute],
        inputs: torch.Tensor,
        dst: int,
        which_half: str = None,
        supported_capacities: Union[List[int], int] = None,
        capacity_padding: bool = False,
        path_wise_padding: bool = False,
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

        model = (
            Model(
                gate=gate,
                supported_capacities=supported_capacities,
                capacity_padding=capacity_padding,
                path_wise_padding=path_wise_padding,
            )
            .eval()
            .cuda()
        )
        for _ in range(5):
            results = model(inputs)
            # print(results)
            if which_half is None:
                real_load = inputs.size(0)
                if capacity_padding:
                    if path_wise_padding:
                        real_load = supported_capacities[dst]
                    else:
                        for c in supported_capacities:
                            if real_load < c:
                                real_load = c
                                break
                self.assertTrue(results[dst].data.size(0) == real_load)
                self.assertTrue(
                    torch.allclose(results[dst].data[: inputs.size(0)], inputs)
                )
                self.assertTrue(results[1 - dst].data.numel() == 0)
                self.assertTrue(torch.allclose(results[2].data, inputs))
            else:
                self.assertTrue(results[1 - dst].data.numel() == 0)
                real_load = inputs.size(0) // 2
                if capacity_padding:
                    if path_wise_padding:
                        real_load = supported_capacities[dst]
                    else:
                        for c in supported_capacities:
                            if real_load < c:
                                real_load = c
                                break
                if which_half == "upper":
                    self.assertTrue(results[dst].data.size(0) == real_load)
                    self.assertTrue(
                        torch.allclose(
                            results[dst].data[: inputs.size(0) // 2],
                            inputs[inputs.size(0) // 2 :],
                        )
                    )
                    self.assertTrue(
                        torch.allclose(results[2].data, inputs[inputs.size(0) // 2 :])
                    )
                elif which_half == "lower":
                    self.assertTrue(results[dst].data.size(0) == real_load)
                    self.assertTrue(
                        torch.allclose(
                            results[dst].data[: inputs.size(0) // 2],
                            inputs[: inputs.size(0) // 2],
                        )
                    )
                    self.assertTrue(
                        torch.allclose(results[2].data, inputs[: inputs.size(0) // 2])
                    )

    def test_padded_route(self):
        multi_dims_sample = [
            torch.arange(0, 80, dtype=torch.float32).view(8, 10).cuda(),
            torch.arange(0, 160, dtype=torch.float32).view(8, 2, 10).cuda(),
            torch.arange(0, 320, dtype=torch.float32).view(8, 2, 10, 2).cuda(),
        ]
        for x in multi_dims_sample:
            for i in range(2):
                supported_capacities = [10, 12]
                self.padded_route(
                    BranchRoute, x, dst=i, supported_capacities=supported_capacities
                )
                self.padded_route(
                    BranchRoute,
                    x,
                    dst=i,
                    supported_capacities=supported_capacities,
                    capacity_padding=True,
                )
                self.padded_route(
                    BranchRoute,
                    x,
                    dst=i,
                    supported_capacities=supported_capacities,
                    capacity_padding=True,
                    path_wise_padding=True,
                )
                self.padded_route(
                    BranchRoute,
                    x,
                    dst=i,
                    which_half="upper",
                    supported_capacities=supported_capacities,
                )
                self.padded_route(
                    BranchRoute,
                    x,
                    dst=i,
                    which_half="upper",
                    supported_capacities=supported_capacities,
                    capacity_padding=True,
                )
                self.padded_route(
                    BranchRoute,
                    x,
                    dst=i,
                    which_half="upper",
                    supported_capacities=supported_capacities,
                    capacity_padding=True,
                    path_wise_padding=True,
                )
                self.padded_route(
                    BranchRoute,
                    x,
                    dst=i,
                    which_half="lower",
                    supported_capacities=supported_capacities,
                )
                self.padded_route(
                    BranchRoute,
                    x,
                    dst=i,
                    which_half="lower",
                    supported_capacities=supported_capacities,
                    capacity_padding=True,
                )
                self.padded_route(
                    BranchRoute,
                    x,
                    dst=i,
                    which_half="lower",
                    supported_capacities=supported_capacities,
                    capacity_padding=True,
                    path_wise_padding=True,
                )

    def test_trace(self):
        gate = nn.Sequential(nn.Linear(10, 2), nn.Softmax(dim=1))

        def jit_script(Model):
            model = Model(gate=gate)
            try:
                graph_module = GraphTracer().trace(model)
                # TODO: add more tests
            except Exception as e:
                self.fail(f"Failed to trace the graph through torch.fx: {e}")

        jit_script(BranchRoute)


if __name__ == "__main__":
    unittest.main()
