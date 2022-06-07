import unittest

import numpy as np
import torch
from brt.router.flow_tensor import (
    FlowTensor,
    deinit_flow_tensor,
    init_flow_tensor,
    make_flow_tensor_cls,
    reset_flow_tensor_cls,
)


class FlowTensorTest(unittest.TestCase):
    def test_runtime_flow_tensor_cls_modification(self):
        make_flow_tensor_cls(extra_attrs=["dummy_a"], default_values=[0])
        from brt.router.flow_tensor import FlowTensor

        self.assertEqual(FlowTensor.EXTRA_ATTRS, ["dummy_a"])
        self.assertEqual(FlowTensor.EXTRA_ATTRS_DEFAULT_VALUES["dummy_a"], 0)
        self.assertEqual(FlowTensor.EXTRA_ATTRS_STACK, ["dummy_a_stack"])
        self.assertTrue(
            hasattr(FlowTensor, "dummy_a") and hasattr(FlowTensor, "dummy_a_stack")
        )

        make_flow_tensor_cls(extra_attrs=["dummy_b"], default_values=[1])
        from brt.router.flow_tensor import FlowTensor

        self.assertEqual(FlowTensor.EXTRA_ATTRS, ["dummy_b"])
        self.assertEqual(FlowTensor.EXTRA_ATTRS_DEFAULT_VALUES["dummy_b"], 1)
        self.assertEqual(FlowTensor.EXTRA_ATTRS_STACK, ["dummy_b_stack"])
        self.assertTrue(
            hasattr(FlowTensor, "dummy_b") and hasattr(FlowTensor, "dummy_b_stack")
        )
        self.assertTrue(
            not hasattr(FlowTensor, "dummy_a")
            and not hasattr(FlowTensor, "dummy_a_stack")
        )

        reset_flow_tensor_cls()
        from brt.router.flow_tensor import FlowTensor

        self.assertEqual(FlowTensor.EXTRA_ATTRS, [])
        self.assertEqual(FlowTensor.EXTRA_ATTRS_DEFAULT_VALUES, {})
        self.assertEqual(FlowTensor.EXTRA_ATTRS_STACK, [])
        self.assertTrue(
            not hasattr(FlowTensor, "dummy_a")
            and not hasattr(FlowTensor, "dummy_a_stack")
        )
        self.assertTrue(
            not hasattr(FlowTensor, "dummy_b")
            and not hasattr(FlowTensor, "dummy_b_stack")
        )

        make_flow_tensor_cls(extra_attrs=["dummy"], default_values=[0])
        flow_tensor = init_flow_tensor(torch.Tensor([1, 2, 3]))
        flow_tensor.pack(torch.tensor([0, 1, 2]), 3)
        self.assertEqual(flow_tensor.dummy, 0)
        flow_tensor.pack(torch.tensor([0, 1, 2, 3]), 4, dummy=1)
        self.assertEqual(flow_tensor.dummy, 1)

    def test_initiliaze(self):
        reset_flow_tensor_cls()
        from brt.router.flow_tensor import FlowTensor

        common_tensor = torch.Tensor([1, 2, 3])
        flow_tensor = torch.Tensor.as_subclass(common_tensor, FlowTensor)
        self.assertFalse(flow_tensor.flow_initilized)
        flow_tensor.pack(torch.tensor([0, 1, 2]), 3)
        assert flow_tensor.flow_initilized is True

    def test_flow_empty(self):
        reset_flow_tensor_cls()
        flow_tensor = init_flow_tensor(torch.Tensor([1, 2, 3]))
        self.assertTrue(flow_tensor.flow_empty)

    def test_pack_unpack(self):
        reset_flow_tensor_cls()
        flow_tensor = init_flow_tensor(torch.Tensor([1, 2, 3]))
        flow_tensor.pack(torch.tensor([0, 1, 2]), 3)
        self.assertTrue(torch.allclose(flow_tensor.tag, torch.tensor([0, 1, 2])))
        self.assertTrue(flow_tensor.load == 3)
        flow_tensor.pack(torch.tensor([0, 1, 2, 3]), 4)
        self.assertTrue(torch.allclose(flow_tensor.tag, torch.tensor([0, 1, 2, 3])))
        self.assertTrue(flow_tensor.load == 4)
        self.assertTrue(flow_tensor.stack_size == 2)
        data, tag, load, _ = flow_tensor.unpack()
        self.assertTrue(torch.allclose(flow_tensor.tag, torch.tensor([0, 1, 2])))
        self.assertTrue(flow_tensor.load == 3)
        self.assertTrue(flow_tensor.stack_size == 1)
    
    def test_deep_pack_deep_unpack(self):
        pass
    
    def test_pack_ret(self):
        pass
    
    def test_collect_attr(self):
        pass
    