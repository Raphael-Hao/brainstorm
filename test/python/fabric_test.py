# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import torch
from brt.router.fabric.zero_skip import ZeroSkipFabric


class FabricTest(unittest.TestCase):
    def test_zero_skip(self):
        # test with all zero tensors
        fabric = ZeroSkipFabric(flow_num=1)
        empty_flow, out_flows = fabric(
            torch.zeros((0, 1, 2, 3), device="cuda"), torch.zeros((0, 4), device="cuda")
        )
        self.assertTrue(empty_flow)
        for i in range(4):
            self.assertEqual(out_flows[i].shape, (0, 1, 2, 3))
            self.assertEqual(out_flows[i].is_cuda, True)
        fabric = ZeroSkipFabric(flow_num=2)
        empty_flow, out_flows = fabric(
            [torch.zeros((0, 1, 2, 3), device="cuda") for _ in range(2)],
            torch.zeros((0, 4), device="cuda"),
        )
        self.assertTrue(empty_flow)
        self.assertEqual(len(out_flows), 2)
        for out_flow in out_flows:
            for i in range(4):
                self.assertEqual(out_flow[i].shape, (0, 1, 2, 3))
                self.assertEqual(out_flow[i].is_cuda, True)

        path_num = 4
        fabric = ZeroSkipFabric(flow_num=1)
        empty_flow, out_flows = fabric(
            [torch.zeros((0, 1, 2, 3), device="cuda") for _ in range(path_num)]
        )
        # print(empty_flow, out_flows)
        self.assertTrue(empty_flow)
        self.assertEqual(out_flows.shape, (0, 1, 2, 3))
        self.assertEqual(out_flows.is_cuda, True)

        fabric = ZeroSkipFabric(flow_num=2)
        empty_flow, out_flows = fabric(
            [
                [torch.zeros((0, 1, 2, 3), device="cuda") for _ in range(path_num)]
                for _ in range(2)
            ]
        )
        # print(empty_flow, out_flows)
        self.assertTrue(empty_flow)
        self.assertEqual(len(out_flows), 2)
        for out_flow in out_flows:
            self.assertEqual(out_flow.shape, (0, 1, 2, 3))
            self.assertEqual(out_flow.is_cuda, True)

        # test non-zero tensors
        fabric = ZeroSkipFabric(flow_num=1)
        empty_flow, out_flows = fabric(
            torch.zeros((1, 1, 2, 3), device="cuda"), torch.zeros((1, 4), device="cuda")
        )
        self.assertFalse(empty_flow)
        self.assertIsNone(out_flows)

        fabric = ZeroSkipFabric(flow_num=2)
        empty_flow, out_flows = fabric(
            [torch.zeros((1, 1, 2, 3), device="cuda") for _ in range(2)],
            torch.zeros((1, 4), device="cuda"),
        )
        self.assertFalse(empty_flow)
        self.assertIsNone(out_flows)

        path_num = 4
        fabric = ZeroSkipFabric(flow_num=1)
        in_flows = [torch.zeros((0, 1, 2, 3), device="cuda") for _ in range(path_num)]
        in_flows[1] = torch.zeros((1, 1, 2, 3), device="cuda")
        empty_flow, out_flows = fabric(in_flows)
        # print(empty_flow, out_flows)
        self.assertFalse(empty_flow)
        self.assertIsNone(out_flows)

        fabric = ZeroSkipFabric(flow_num=2)
        in_flows = [
            [torch.zeros((0, 1, 2, 3), device="cuda") for _ in range(path_num)]
            for _ in range(2)
        ]
        in_flows[0][1] = torch.zeros((1, 1, 2, 3), device="cuda")
        in_flows[1][1] = torch.zeros((1, 1, 2, 3), device="cuda")

        empty_flow, out_flows = fabric(in_flows)
        # print(empty_flow, out_flows)
        self.assertFalse(empty_flow)
        self.assertIsNone(out_flows)

if __name__ == "__main__":
    unittest.main()
