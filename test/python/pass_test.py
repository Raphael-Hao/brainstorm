# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import numpy as np

import torch
from torch import nn
from torch.fx import GraphModule, Graph, Node

import brt
from brt.runtime.log import set_level_to_debug
from brt.trace.graph import symbolic_trace
from brt.passes import VerticalFusePass, RouterFixPass, HorizFusePass
from brt.router import ScatterRouter, GatherRouter


class PassTest(unittest.TestCase):
    def test_vertical_fuse_pass(self):
        class TestModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.router = ScatterRouter(capturing=True, capture_mode="max")
                self.expert0 = nn.Sequential(
                    nn.Conv2d(8, 8, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(8, 8, kernel_size=3, padding=1),
                )
                self.expert1 = nn.Sequential(
                    nn.Conv2d(8, 8, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(8, 8, kernel_size=3, padding=1),
                )
                self.gather = GatherRouter()

            def forward(self, x: torch.Tensor):
                # x: [4, 8, 32, 32]
                # return: [4, 8, 32, 32]
                mean = torch.mean(x, dim=(1, 2, 3)).unsqueeze(1)
                scores = torch.cat((mean, torch.zeros_like(mean)), dim=1)
                print(scores)
                y0, y1 = self.router(x, scores)
                print(y0.shape)
                print(y0.tag)
                print(y1.shape)
                print(y1.tag)
                y0 = self.expert0(y0)
                y1 = self.expert1(y1)
                y = self.gather([y0, y1])
                return y

        set_level_to_debug()

        testm = TestModule().eval().cuda()
        input_tensor = torch.randn(4, 8, 32, 32, requires_grad=False).cuda()
        testm.router.load_history = np.array([4, 4], dtype=int)
        raw_output_tensor = testm(input_tensor)
        print(raw_output_tensor.shape)

        testm_trace = symbolic_trace(testm)
        print(testm_trace.graph)

        router_fix_pass = RouterFixPass(testm)
        router_fix_pass.run_on_graph()
        testm_rf = router_fix_pass.finalize()

        vertical_fuse_pass = VerticalFusePass(
            testm_rf, sample_inputs={"x": input_tensor}
        )
        vertical_fuse_pass.run_on_graph()
        testm_vf = vertical_fuse_pass.finalize()
        print(testm_vf.graph)

        vf_output_tensor = testm_vf(input_tensor)
        print(vf_output_tensor.shape)

        # print(
        #     torch.cat(
        #         [
        #             (raw_output_tensor - vf_output_tensor).unsqueeze(4),
        #             raw_output_tensor.unsqueeze(4),
        #         ],
        #         dim=-1,
        #     )
        # )

        assert torch.allclose(raw_output_tensor, vf_output_tensor, rtol=1e-3, atol=1e-4)

    def test_horiz_fuse_pass(self):
        class TestModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.router = ScatterRouter(capturing=True, capture_mode="max")
                self.expert0 = nn.Sequential(
                    nn.Conv2d(8, 8, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(8, 8, kernel_size=3, padding=1),
                )
                self.expert1 = nn.Sequential(
                    nn.Conv2d(8, 8, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(8, 8, kernel_size=3, padding=1),
                )
                self.gather = GatherRouter()

            def forward(self, x: torch.Tensor):
                # x: [4, 8, 32, 32]
                # return: [4, 8, 32, 32]
                mean = torch.mean(x, dim=(1, 2, 3)).unsqueeze(1)
                scores = torch.cat((mean, torch.zeros_like(mean)), dim=1)
                print(scores)
                y0, y1 = self.router(x, scores)
                print(y0.shape)
                print(y0.tag)
                print(y1.shape)
                print(y1.tag)
                y0 = self.expert0(y0)
                y1 = self.expert1(y1)
                y = self.gather([y0, y1])
                return y

        set_level_to_debug()

        testm = TestModule().eval().cuda()
        input_tensor = torch.randn(4, 8, 32, 32, requires_grad=False).cuda()
        testm.router.load_history = np.array([4, 4], dtype=int)
        raw_output_tensor = testm(input_tensor)
        print(raw_output_tensor.shape)

        testm_trace = symbolic_trace(testm)
        print(testm_trace.graph)

        router_fix_pass = RouterFixPass(testm)
        router_fix_pass.run_on_graph()
        testm_rf = router_fix_pass.finalize()

        horiz_fuse_pass = HorizFusePass(
            testm_rf, sample_inputs={"x": input_tensor}
        )
        horiz_fuse_pass.run_on_graph()
        testm_hf = horiz_fuse_pass.finalize()
        print(testm_hf.graph)

        hf_output_tensor = testm_hf(input_tensor)
        print(hf_output_tensor.shape)

        print(
            torch.cat(
                [
                    (raw_output_tensor - hf_output_tensor).unsqueeze(4),
                    raw_output_tensor.unsqueeze(4),
                ],
                dim=-1,
            )
        )

        assert torch.allclose(raw_output_tensor, hf_output_tensor, rtol=1e-3, atol=1e-4)
