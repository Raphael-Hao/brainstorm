import unittest

import numpy as np
import torch
from brt.router.proto_tensor import (
    collect_proto_attr_stack,
    deinit_proto_tensor,
    init_proto_tensor,
    make_proto_tensor_cls,
    pack_proto_attr_stack,
    reset_proto_tensor_cls,
)


class ProtoTensorTest(unittest.TestCase):
    def test_runtime_flow_tensor_cls_modification(self):
        make_proto_tensor_cls(extra_attrs=["dummy_a"], default_values=[0], mode="new")
        from brt.router.proto_tensor import ProtoTensor

        self.assertEqual(ProtoTensor.EXTRA_ATTRS, ["dummy_a"])
        self.assertEqual(ProtoTensor.EXTRA_ATTRS_DEFAULT_VALUES["dummy_a"], 0)
        self.assertEqual(ProtoTensor.EXTRA_ATTRS_STACK, ["dummy_a_stack"])
        self.assertTrue(
            hasattr(ProtoTensor, "dummy_a") and hasattr(ProtoTensor, "dummy_a_stack")
        )

        make_proto_tensor_cls(extra_attrs=["dummy_b"], default_values=[1], mode="new")
        from brt.router.proto_tensor import ProtoTensor

        self.assertEqual(ProtoTensor.EXTRA_ATTRS, ["dummy_b"])
        self.assertEqual(ProtoTensor.EXTRA_ATTRS_DEFAULT_VALUES["dummy_b"], 1)
        self.assertEqual(ProtoTensor.EXTRA_ATTRS_STACK, ["dummy_b_stack"])
        self.assertTrue(
            hasattr(ProtoTensor, "dummy_b") and hasattr(ProtoTensor, "dummy_b_stack")
        )
        self.assertTrue(
            not hasattr(ProtoTensor, "dummy_a")
            and not hasattr(ProtoTensor, "dummy_a_stack")
        )

        reset_proto_tensor_cls()
        from brt.router.proto_tensor import ProtoTensor

        self.assertEqual(ProtoTensor.EXTRA_ATTRS, [])
        self.assertEqual(ProtoTensor.EXTRA_ATTRS_DEFAULT_VALUES, {})
        self.assertEqual(ProtoTensor.EXTRA_ATTRS_STACK, [])
        self.assertTrue(
            not hasattr(ProtoTensor, "dummy_a")
            and not hasattr(ProtoTensor, "dummy_a_stack")
        )
        self.assertTrue(
            not hasattr(ProtoTensor, "dummy_b")
            and not hasattr(ProtoTensor, "dummy_b_stack")
        )

    def test_initiliaze(self):
        reset_proto_tensor_cls()
        from brt.router.proto_tensor import ProtoTensor

        common_tensor = torch.Tensor([1, 2, 3])
        proto_tensor = torch.Tensor.as_subclass(common_tensor, ProtoTensor)
        self.assertFalse(proto_tensor.proto_initilized)
        proto_tensor.pack(torch.tensor([0, 1, 2]), 3)
        assert proto_tensor.proto_initilized is True

        make_proto_tensor_cls(extra_attrs=["dummy"], default_values=[0], mode="new")
        proto_tensor = init_proto_tensor(torch.Tensor([1, 2, 3]))
        proto_tensor.pack(torch.tensor([0, 1, 2]), 3)
        self.assertEqual(proto_tensor.dummy, 0)
        proto_tensor.pack(torch.tensor([0, 1, 2, 3]), 4, dummy=1)
        self.assertEqual(proto_tensor.dummy, 1)
        data, _, _, _ = deinit_proto_tensor(proto_tensor)
        self.assertIsInstance(data, torch.Tensor)
        self.assertNotIsInstance(data, ProtoTensor)

    def test_flow_empty(self):
        reset_proto_tensor_cls()
        proto_tensor = init_proto_tensor(torch.Tensor([1, 2, 3]))
        self.assertTrue(proto_tensor.proto_empty)

    def test_pack_unpack(self):
        reset_proto_tensor_cls()
        proto_tensor = init_proto_tensor(torch.Tensor([1, 2, 3]))
        proto_tensor.pack(torch.tensor([0, 1, 2]), 3)
        self.assertTrue(torch.allclose(proto_tensor.tag, torch.tensor([0, 1, 2])))
        self.assertEqual(proto_tensor.load, 3)
        proto_tensor.pack(torch.tensor([0, 1, 2, 3]), 4)
        self.assertTrue(torch.allclose(proto_tensor.tag, torch.tensor([0, 1, 2, 3])))
        self.assertEqual(proto_tensor.load, 4)
        self.assertEqual(proto_tensor.stack_size, 2)
        _, _, _, _ = proto_tensor.unpack()
        self.assertTrue(torch.allclose(proto_tensor.tag, torch.tensor([0, 1, 2])))
        self.assertTrue(proto_tensor.load == 3)
        self.assertTrue(proto_tensor.stack_size == 1)

    def test_deep_pack_deep_unpack(self):
        reset_proto_tensor_cls()
        proto_tensor = init_proto_tensor(torch.Tensor([1, 2, 3]))
        proto_tensor.pack(torch.Tensor([0, 1]), 2)
        tag_stack = [torch.Tensor([0, 1, 2]), torch.Tensor([0, 1, 2, 3])]
        load_stack = [3, 4]
        proto_tensor.deep_pack(tag_stack, load_stack)
        self.assertEqual(proto_tensor.stack_size, 2)
        self.assertEqual(proto_tensor.load, 4)
        self.assertTrue(torch.allclose(proto_tensor.tag, torch.Tensor([0, 1, 2, 3])))
        (
            unpack_flow_tensor,
            unpack_tag_stack,
            unpack_load_stack,
            _,
        ) = proto_tensor.deep_unpack()
        self.assertEqual(unpack_flow_tensor.stack_size, 0)
        self.assertEqual(unpack_load_stack, load_stack)
        self.assertEqual(unpack_tag_stack, tag_stack)
        self.assertEqual(id(proto_tensor), id(unpack_flow_tensor))

    def test_pack_ret(self):
        reset_proto_tensor_cls()
        tag_stack = [torch.Tensor([0, 1, 2]), torch.Tensor([0, 1, 2, 3])]
        load_stack = [3, 4]
        proto_tensor = init_proto_tensor(torch.Tensor([1, 2, 3]))
        proto_tensor = pack_proto_attr_stack(proto_tensor, tag_stack, load_stack)
        self.assertEqual(proto_tensor.stack_size, 2)
        self.assertEqual(proto_tensor.load, 4)
        self.assertTrue(torch.allclose(proto_tensor.tag, torch.Tensor([0, 1, 2, 3])))

        proto_tensor_list = [
            init_proto_tensor(torch.Tensor([1, 2, 3])),
            init_proto_tensor(torch.Tensor([1, 2, 3])),
        ]
        proto_tensor_list = pack_proto_attr_stack(
            proto_tensor_list, tag_stack, load_stack
        )
        for i in range(2):
            self.assertEqual(proto_tensor_list[i].stack_size, 2)
            self.assertEqual(proto_tensor_list[i].load, 4)
            self.assertTrue(
                torch.allclose(proto_tensor_list[i].tag, torch.Tensor([0, 1, 2, 3]))
            )

        proto_tensor_mix_list = [
            init_proto_tensor(torch.Tensor([1, 2, 3])),
            [
                init_proto_tensor(torch.Tensor([1, 2, 3])),
                init_proto_tensor(torch.Tensor([1, 2, 3])),
            ],
        ]
        proto_tensor_mix_list = pack_proto_attr_stack(
            proto_tensor_mix_list, tag_stack, load_stack
        )
        self.assertEqual(proto_tensor_mix_list[0].stack_size, 2)
        self.assertEqual(proto_tensor_mix_list[0].load, 4)
        self.assertTrue(
            torch.allclose(proto_tensor_mix_list[0].tag, torch.Tensor([0, 1, 2, 3]))
        )
        for i in range(2):
            self.assertEqual(proto_tensor_mix_list[1][i].stack_size, 2)
            self.assertEqual(proto_tensor_mix_list[1][i].load, 4)
            self.assertTrue(
                torch.allclose(
                    proto_tensor_mix_list[1][i].tag, torch.Tensor([0, 1, 2, 3])
                )
            )

        class dummy:
            pass

        proto_tensor_dummy_mix_list = [
            dummy(),
            [
                init_proto_tensor(torch.Tensor([1, 2, 3])),
                init_proto_tensor(torch.Tensor([1, 2, 3])),
            ],
        ]
        proto_tensor_dummy_mix_list = pack_proto_attr_stack(
            proto_tensor_dummy_mix_list, tag_stack, load_stack
        )
        self.assertTrue(hasattr(proto_tensor_mix_list[0], "stack_size"))
        self.assertTrue(hasattr(proto_tensor_mix_list[0], "load"))
        self.assertTrue(hasattr(proto_tensor_mix_list[0], "tag"))
        for i in range(2):
            self.assertEqual(proto_tensor_mix_list[1][i].stack_size, 2)
            self.assertEqual(proto_tensor_mix_list[1][i].load, 4)
            self.assertTrue(
                torch.allclose(
                    proto_tensor_mix_list[1][i].tag, torch.Tensor([0, 1, 2, 3])
                )
            )

    def test_collect_attr(self):
        reset_proto_tensor_cls()
        tag_stack = [torch.Tensor([0, 1, 2]), torch.Tensor([0, 1, 2, 3])]
        load_stack = [3, 4]
        proto_tensor = init_proto_tensor(torch.Tensor([1, 2, 3]))
        proto_tensor = pack_proto_attr_stack(proto_tensor, tag_stack, load_stack)
        (
            collected_tag_stack,
            collected_load_stack,
            collected_extra_attr_stack_dict,
        ) = collect_proto_attr_stack(proto_tensor)
        self.assertEqual(collected_tag_stack, tag_stack)
        self.assertEqual(collected_load_stack, load_stack)
        self.assertEqual(collected_extra_attr_stack_dict, {})

        proto_tensor_mix_list = [
            init_proto_tensor(torch.Tensor([1, 2, 3])),
            [
                init_proto_tensor(torch.Tensor([1, 2, 3])),
                init_proto_tensor(torch.Tensor([1, 2, 3])),
            ],
        ]
        proto_tensor_mix_list = pack_proto_attr_stack(
            proto_tensor_mix_list, tag_stack, load_stack
        )
        (
            collected_tag_stack,
            collected_load_stack,
            collected_extra_attr_stack_dict,
        ) = collect_proto_attr_stack(proto_tensor_mix_list)
        self.assertEqual(collected_tag_stack, tag_stack)
        self.assertEqual(collected_load_stack, load_stack)
        self.assertEqual(collected_extra_attr_stack_dict, {})

        class dummy:
            pass

        proto_tensor_dummy_mix_list = [
            dummy(),
            [
                init_proto_tensor(torch.Tensor([1, 2, 3])),
                init_proto_tensor(torch.Tensor([1, 2, 3])),
            ],
        ]
        proto_tensor_dummy_mix_list = pack_proto_attr_stack(
            proto_tensor_dummy_mix_list, tag_stack, load_stack
        )
        (
            collected_tag_stack,
            collected_load_stack,
            collected_extra_attr_stack_dict,
        ) = collect_proto_attr_stack(proto_tensor_mix_list)
        self.assertEqual(collected_tag_stack, tag_stack)
        self.assertEqual(collected_load_stack, load_stack)
        self.assertEqual(collected_extra_attr_stack_dict, {})


if __name__ == "__main__":
    unittest.main()
