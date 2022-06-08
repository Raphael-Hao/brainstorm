import unittest

import numpy as np
import torch
from brt.router.flow_tensor import (
    FlowTensor,
    collect_attr_stack,
    deinit_flow_tensor,
    init_flow_tensor,
    make_flow_tensor_cls,
    pack_ret,
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

    def test_initiliaze(self):
        reset_flow_tensor_cls()
        from brt.router.flow_tensor import FlowTensor

        common_tensor = torch.Tensor([1, 2, 3])
        flow_tensor = torch.Tensor.as_subclass(common_tensor, FlowTensor)
        self.assertFalse(flow_tensor.flow_initilized)
        flow_tensor.pack(torch.tensor([0, 1, 2]), 3)
        assert flow_tensor.flow_initilized is True

        make_flow_tensor_cls(extra_attrs=["dummy"], default_values=[0])
        flow_tensor = init_flow_tensor(torch.Tensor([1, 2, 3]))
        flow_tensor.pack(torch.tensor([0, 1, 2]), 3)
        self.assertEqual(flow_tensor.dummy, 0)
        flow_tensor.pack(torch.tensor([0, 1, 2, 3]), 4, dummy=1)
        self.assertEqual(flow_tensor.dummy, 1)
        data, _, _, _ = deinit_flow_tensor(flow_tensor)
        self.assertIsInstance(data, torch.Tensor)
        self.assertNotIsInstance(data, FlowTensor)

    def test_flow_empty(self):
        reset_flow_tensor_cls()
        flow_tensor = init_flow_tensor(torch.Tensor([1, 2, 3]))
        self.assertTrue(flow_tensor.flow_empty)

    def test_pack_unpack(self):
        reset_flow_tensor_cls()
        flow_tensor = init_flow_tensor(torch.Tensor([1, 2, 3]))
        flow_tensor.pack(torch.tensor([0, 1, 2]), 3)
        self.assertTrue(torch.allclose(flow_tensor.tag, torch.tensor([0, 1, 2])))
        self.assertEqual(flow_tensor.load, 3)
        flow_tensor.pack(torch.tensor([0, 1, 2, 3]), 4)
        self.assertTrue(torch.allclose(flow_tensor.tag, torch.tensor([0, 1, 2, 3])))
        self.assertEqual(flow_tensor.load, 4)
        self.assertEqual(flow_tensor.stack_size, 2)
        data, tag, load, _ = flow_tensor.unpack()
        self.assertTrue(torch.allclose(flow_tensor.tag, torch.tensor([0, 1, 2])))
        self.assertTrue(flow_tensor.load == 3)
        self.assertTrue(flow_tensor.stack_size == 1)

    def test_deep_pack_deep_unpack(self):
        reset_flow_tensor_cls()
        flow_tensor = init_flow_tensor(torch.Tensor([1, 2, 3]))
        flow_tensor.pack(torch.Tensor([0, 1]), 2)
        tag_stack = [torch.Tensor([0, 1, 2]), torch.Tensor([0, 1, 2, 3])]
        load_stack = [3, 4]
        flow_tensor.deep_pack(tag_stack, load_stack)
        self.assertEqual(flow_tensor.stack_size, 2)
        self.assertEqual(flow_tensor.load, 4)
        self.assertTrue(torch.allclose(flow_tensor.tag, torch.Tensor([0, 1, 2, 3])))
        (
            unpack_flow_tensor,
            unpack_tag_stack,
            unpack_load_stack,
            _,
        ) = flow_tensor.deep_unpack()
        self.assertEqual(unpack_flow_tensor.stack_size, 0)
        self.assertEqual(unpack_load_stack, load_stack)
        self.assertEqual(unpack_tag_stack, tag_stack)
        self.assertEqual(id(flow_tensor), id(unpack_flow_tensor))

    def test_pack_ret(self):
        reset_flow_tensor_cls()
        tag_stack = [torch.Tensor([0, 1, 2]), torch.Tensor([0, 1, 2, 3])]
        load_stack = [3, 4]
        flow_tensor = init_flow_tensor(torch.Tensor([1, 2, 3]))
        flow_tensor = pack_ret(flow_tensor, tag_stack, load_stack)
        self.assertEqual(flow_tensor.stack_size, 2)
        self.assertEqual(flow_tensor.load, 4)
        self.assertTrue(torch.allclose(flow_tensor.tag, torch.Tensor([0, 1, 2, 3])))

        flow_tensor_list = [
            init_flow_tensor(torch.Tensor([1, 2, 3])),
            init_flow_tensor(torch.Tensor([1, 2, 3])),
        ]
        flow_tensor_list = pack_ret(flow_tensor_list, tag_stack, load_stack)
        for i in range(2):
            self.assertEqual(flow_tensor_list[i].stack_size, 2)
            self.assertEqual(flow_tensor_list[i].load, 4)
            self.assertTrue(
                torch.allclose(flow_tensor_list[i].tag, torch.Tensor([0, 1, 2, 3]))
            )

        flow_tensor_mix_list = [
            init_flow_tensor(torch.Tensor([1, 2, 3])),
            [
                init_flow_tensor(torch.Tensor([1, 2, 3])),
                init_flow_tensor(torch.Tensor([1, 2, 3])),
            ],
        ]
        flow_tensor_mix_list = pack_ret(flow_tensor_mix_list, tag_stack, load_stack)
        self.assertEqual(flow_tensor_mix_list[0].stack_size, 2)
        self.assertEqual(flow_tensor_mix_list[0].load, 4)
        self.assertTrue(
            torch.allclose(flow_tensor_mix_list[0].tag, torch.Tensor([0, 1, 2, 3]))
        )
        for i in range(2):
            self.assertEqual(flow_tensor_mix_list[1][i].stack_size, 2)
            self.assertEqual(flow_tensor_mix_list[1][i].load, 4)
            self.assertTrue(
                torch.allclose(
                    flow_tensor_mix_list[1][i].tag, torch.Tensor([0, 1, 2, 3])
                )
            )

        class dummy:
            pass

        flow_tensor_dummy_mix_list = [
            dummy(),
            [
                init_flow_tensor(torch.Tensor([1, 2, 3])),
                init_flow_tensor(torch.Tensor([1, 2, 3])),
            ],
        ]
        flow_tensor_dummy_mix_list = pack_ret(
            flow_tensor_dummy_mix_list, tag_stack, load_stack
        )
        self.assertTrue(hasattr(flow_tensor_mix_list[0], "stack_size"))
        self.assertTrue(hasattr(flow_tensor_mix_list[0], "load"))
        self.assertTrue(hasattr(flow_tensor_mix_list[0], "tag"))
        for i in range(2):
            self.assertEqual(flow_tensor_mix_list[1][i].stack_size, 2)
            self.assertEqual(flow_tensor_mix_list[1][i].load, 4)
            self.assertTrue(
                torch.allclose(
                    flow_tensor_mix_list[1][i].tag, torch.Tensor([0, 1, 2, 3])
                )
            )

    def test_collect_attr(self):
        reset_flow_tensor_cls()
        tag_stack = [torch.Tensor([0, 1, 2]), torch.Tensor([0, 1, 2, 3])]
        load_stack = [3, 4]
        flow_tensor = init_flow_tensor(torch.Tensor([1, 2, 3]))
        flow_tensor = pack_ret(flow_tensor, tag_stack, load_stack)
        collected_tag_stack, collected_load_stack, collected_extra_attr_stack_dict = collect_attr_stack(flow_tensor)
        self.assertEqual(collected_tag_stack, tag_stack)
        self.assertEqual(collected_load_stack, load_stack)
        self.assertEqual(collected_extra_attr_stack_dict, {})
        
        flow_tensor_mix_list = [
            init_flow_tensor(torch.Tensor([1, 2, 3])),
            [
                init_flow_tensor(torch.Tensor([1, 2, 3])),
                init_flow_tensor(torch.Tensor([1, 2, 3])),
            ],
        ]
        flow_tensor_mix_list = pack_ret(flow_tensor_mix_list, tag_stack, load_stack)
        collected_tag_stack, collected_load_stack, collected_extra_attr_stack_dict = collect_attr_stack(flow_tensor_mix_list)
        self.assertEqual(collected_tag_stack, tag_stack)
        self.assertEqual(collected_load_stack, load_stack)
        self.assertEqual(collected_extra_attr_stack_dict, {})
        
        class dummy:
            pass

        flow_tensor_dummy_mix_list = [
            dummy(),
            [
                init_flow_tensor(torch.Tensor([1, 2, 3])),
                init_flow_tensor(torch.Tensor([1, 2, 3])),
            ],
        ]
        flow_tensor_dummy_mix_list = pack_ret(
            flow_tensor_dummy_mix_list, tag_stack, load_stack
        )
        collected_tag_stack, collected_load_stack, collected_extra_attr_stack_dict = collect_attr_stack(flow_tensor_mix_list)
        self.assertEqual(collected_tag_stack, tag_stack)
        self.assertEqual(collected_load_stack, load_stack)
        self.assertEqual(collected_extra_attr_stack_dict, {})
        
