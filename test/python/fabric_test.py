# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import torch
from brt.router.fabric.zero_skip import ZeroSkipFabric
from brt._C import router


class FabricCPPTest(unittest.TestCase):
    def generate_mask(self, cell_size, path_num, topk):
        score = torch.randn((cell_size, path_num), device="cuda")
        indices = torch.topk(score, topk, dim=1).indices
        mask = torch.zeros(
            (cell_size, path_num), dtype=torch.int32, device=indices.device
        ).scatter(1, indices, 1)
        return mask

    def brt_generate_indices(
        self,
        mask,
        supported_capacities=None,
        capacity_padding=False,
        is_tag_index=False,
        load_on_cpu=False,
    ):
        return router.generate_indices_and_loads(
            mask, supported_capacities, capacity_padding, is_tag_index, load_on_cpu
        )

    def pt_generate_indices(
        self,
        mask,
        supported_capacities=None,
        capacity_padding=False,
        is_tag_index=False,
        load_on_cpu=False,
    ):
        if not is_tag_index:
            indices = torch.cumsum(mask, dim=0) * mask
            loads = torch.sum(mask, dim=0)
            if supported_capacities is not None:
                for i in range(mask.size(1)):
                    real_load = loads[i]
                    if real_load == 0:
                        continue
                    mapped = False
                    for capacity in supported_capacities:
                        if real_load <= capacity:
                            if capacity_padding:
                                real_load = capacity
                            loads[i] = real_load
                            mapped = True
                            break
                    if not mapped:
                        loads[i] = supported_capacities[-1]
        else:
            indices = torch.zeros_like(mask)
            loads = torch.zeros(mask.size(1), dtype=torch.int32, device=indices.device)
            mask_t = mask.t().contiguous()
            for i in range(mask.size(1)):
                indices_per_path = mask_t[i].view(-1).nonzero() + 1
                loads[i] = indices_per_path.numel()
                indices[: indices_per_path.numel(), i : i + 1] = indices_per_path
                if supported_capacities is not None:
                    real_load = loads[i]
                    if real_load == 0:
                        continue
                    mapped = False
                    for capacity in supported_capacities:
                        if real_load <= capacity:
                            if capacity_padding:
                                real_load = capacity
                            loads[i] = real_load
                            mapped = True
                            break
                    if not mapped:
                        loads[i] = supported_capacities[-1]

        if load_on_cpu:
            loads = loads.cpu()
        return indices.to(torch.int32), loads.to(torch.int32)

    def check_indices(
        self,
        mask,
        supported_capacities=None,
        capacity_padding=False,
        is_tag_index=False,
    ):
        brt_indice, brt_load = self.brt_generate_indices(
            mask, supported_capacities, capacity_padding, is_tag_index
        )
        pt_indice, pt_load = self.pt_generate_indices(
            mask, supported_capacities, capacity_padding, is_tag_index
        )
        self.assertTrue(torch.allclose(brt_indice, pt_indice))
        self.assertTrue(torch.allclose(brt_load, pt_load))

    def test_indices_generation(self):
        # test with topk=1
        cell_num = 4096
        path_num = 16
        mask = self.generate_mask(cell_num, path_num, 1)
        self.check_indices(mask, is_tag_index=False)
        self.check_indices(mask, is_tag_index=True)
        # test with topk=4
        mask = self.generate_mask(cell_num, path_num, 4)
        self.check_indices(mask, is_tag_index=False)
        self.check_indices(mask, is_tag_index=True)

        # test with supported_capacities=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        path_num = 8
        mask = self.generate_mask(cell_num, path_num, 4)
        supported_capacities = torch.tensor(
            [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
            dtype=torch.int32,
            device=mask.device,
        )
        self.check_indices(mask, supported_capacities, is_tag_index=False)
        self.check_indices(mask, supported_capacities, is_tag_index=True)

        # test with capacity_padding=True
        cell_num = 512
        mask = self.generate_mask(cell_num, path_num, 1)
        self.check_indices(
            mask, supported_capacities, capacity_padding=True, is_tag_index=False
        )
        self.check_indices(
            mask, supported_capacities, capacity_padding=True, is_tag_index=True
        )

    def test_dispatch(self):
        cell_num = 4096
        path_num = 16
        input_cells = torch.randn((cell_num, 1, 2, 3), device="cuda")
        mask = self.generate_mask(cell_num, path_num, 1)
        indices, loads = self.brt_generate_indices(mask, is_tag_index=False)
        output_cells = router.dispatch_with_indices_and_loads(
            input_cells, indices, loads
        )
        print(output_cells)

    def test_combine(self):
        pass


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
