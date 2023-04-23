# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import torch
import torch.nn as nn
from brt import Annotator
from brt.runtime.grid_tensor import GridTensor, deinit_grid_tensor, init_grid_tensor


class GridTensorTest(unittest.TestCase):
    def test_annotate(self):
        torch_t = torch.randn(2, 3, 4)
        annotator = Annotator(dims=[0, 1])
        grid_t = annotator(torch_t, dims=[0, 1])
        self.assertEqual(grid_t.shape, torch.Size([6, 4]))
        torch_t = torch.randn(2, 3, 4)
        annotator = Annotator(dims=[2, 1])
        grid_t = annotator(torch_t, dims=[2, 1])
        self.assertEqual(grid_t.shape, torch.Size([12, 2]))
        torch_t = torch.randn(2, 3, 4)
        annotator = Annotator(dims=[2], cell_shape=[2, 3, 2])
        grid_t = annotator(torch_t, dims=[2], cell_shape=[2, 3, 2])
        self.assertEqual(grid_t.shape, torch.Size([2, 2, 3, 2]))

        print(grid_t.load)
        print(grid_t.tag)

    def test_initiliaze(self):
        torch_t = torch.Tensor([1, 2, 3])
        grid_t = torch.Tensor.as_subclass(torch_t, GridTensor)
        self.assertFalse(grid_t.cell_initilized)
        grid_t.pack(torch.tensor([0, 1, 2]), 3)
        self.assertTrue(grid_t.cell_initilized)

        grid_t = init_grid_tensor(torch.Tensor([1, 2, 3]))
        grid_t.pack(torch.tensor([0, 1, 2]), torch.tensor([3]), dummy_a=0)
        self.assertEqual(grid_t.get_extra_attr("dummy_a"), 0)
        grid_t.pack(torch.tensor([0, 1, 2, 3]), 4, dummy_b=1)
        self.assertEqual(grid_t.get_extra_attr("dummy_b"), 1)
        data, _, _, _ = deinit_grid_tensor(grid_t)
        self.assertIsInstance(data, torch.Tensor)
        self.assertNotIsInstance(data, GridTensor)

    def test_torch_function(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(1, 1)

            def forward(self, x):
                return self.linear(x)

        model = Model()
        GridTensor.SHALLOW_TRANSPORT = False
        annotator = Annotator(dims=[0])
        input_x = annotator(torch.Tensor([1]), dims=[0])
        output_x = model(input_x)
        self.assertEqual(input_x.tag_stack, output_x.tag_stack)
        self.assertEqual(input_x.load_stack, output_x.load_stack)
        self.assertNotEqual(id(input_x.tag_stack), id(output_x.tag_stack))
        self.assertNotEqual(id(input_x.load_stack), id(output_x.load_stack))
        GridTensor.SHALLOW_TRANSPORT = True
        annotator = Annotator(dims=[0])
        input_x = annotator(torch.Tensor([1]), dims=[0])
        output_x = model(input_x)
        self.assertEqual(input_x.tag_stack, output_x.tag_stack)
        self.assertEqual(input_x.load_stack, output_x.load_stack)
        self.assertEqual(id(input_x.tag_stack), id(output_x.tag_stack))
        self.assertEqual(id(input_x.load_stack), id(output_x.load_stack))
