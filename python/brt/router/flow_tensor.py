import copy
from typing import Any, Dict, List, Tuple, TypeVar

import torch
from brt.common import log

logger = log.get_logger(__file__)


def flow_attr_factory(name):
    def getter(self):
        return self._get_brt_attr(name)

    def setter(self, value):
        self._set_brt_attr(name, value)

    return property(getter, setter, doc=f"brt FlowTensor attr: {name}")


def flow_attr_stack_factory(name):
    def getter(self):
        return self._get_brt_attr_stack(name)

    def setter(self, value):
        self._set_brt_attr_stack(name, value)

    return property(getter, setter, doc=f"brt FlowTensor attr stack: {name}")


def _make_flow_tensor(extra_attrs: List[str], default_values: List[Any]):
    extra_attrs_set = set(extra_attrs)
    assert (len(extra_attrs_set) == len(extra_attrs)) and (
        len(extra_attrs_set) == len(default_values)
    )
    extra_attrs_stack = [attr + "_stack" for attr in extra_attrs]

    def collect_attr_stack(
        args,
    ) -> Tuple[List[torch.Tensor], List[int], Dict[str, List[Any]]]:
        """collect all attr stack from args
           We have a big assumption here that the args of FlowTensor type will shared the same attr stack.
           Common nn.Module only transmit them the attr stack of FlowTensor without modification.
        Args:
            args (_type_): args of a torch function handled by __torch_function__

        Returns:
            all_attr_stack: including mandatory tag stack, load stack and extra attrs stack
        """
        all_tag_stack = []
        all_load_stack = []
        all_extra_attrs_stack_dict = {
            attr_stack: [] for attr_stack in extra_attrs_stack
        }

        if isinstance(args, FlowTensor):
            all_tag_stack = args.tag_stack
            all_load_stack = args.load_stack
            for attr_stack in extra_attrs_stack:
                all_extra_attrs_stack_dict[attr_stack] = getattr(args, attr_stack)
            return all_tag_stack, all_load_stack, all_extra_attrs_stack_dict

        if isinstance(args, (Tuple, List)):
            for a in args:
                _all_tags, _all_loads, _all_extra_attrs_stack_dict = collect_attr_stack(
                    a
                )
                if _all_tags and _all_loads:
                    return _all_tags, _all_loads, _all_extra_attrs_stack_dict

        return all_tag_stack, all_load_stack, all_extra_attrs_stack_dict

    def pack_ret(
        ret,
        tag_stack: List[torch.Tensor],
        load_stack: List[int],
        extra_attrs_stack_dict: Dict[str, List[Any]],
    ):

        if isinstance(ret, FlowTensor):
            ret = ret.deep_pack(tag_stack, load_stack, **extra_attrs_stack_dict)

        if isinstance(ret, (Tuple, List)):
            ret = type(ret)(
                pack_ret(t, tag_stack, load_stack, extra_attrs_stack_dict) for t in ret
            )

        return ret

    class FlowTensor(torch.Tensor):
        CHECK_TAGS = False
        EXTRA_ATTRS = extra_attrs
        EXTRA_ATTRS_DEFAULT_VALUES = default_values
        EXTRA_ATTRS_STACK = extra_attrs_stack

        def init_flow(self):
            if not self.flow_initilized:
                self.__dict__["brt_tag_stack"] = []
                self.__dict__["brt_load_stack"] = []
                for attr_stack in extra_attrs_stack:
                    self.__dict__["brt_" + attr_stack] = []

        def _get_brt_attr_stack(self, attr_stack):
            return self.__dict__["brt_" + attr_stack]

        def _set_brt_attr_stack(self, attr_stack, value):
            """We need at least a shadow copy here because the attr_stack can be shared with other FlowTensor.
            otherwise, modifying other FlowTensor will modify the attr_stack of this FlowTensor.
            """
            self.__dict__["brt_" + attr_stack] = copy.copy(value)

        def _get_brt_attr(self, attr):
            assert (
                not self.flow_empty()
            ), f"Trying to get {attr} from a FlowTensor without packed tag and load"
            return self.__dict__["brt_" + attr + "_stack"][-1]

        def _set_brt_attr(self, attr, value):
            assert (
                not self.flow_empty()
            ), f"Trying to set {attr} to {value} for a FlowTensor without packed tag and load"
            self.__dict__["brt_" + attr + "_stack"][-1] = value

        def _push_brt_attr(self, attr, value):
            self.__dict__["brt_" + attr + "_stack"].append(value)

        def _pop_brt_attr(self, attr):
            return self.__dict__["brt_" + attr + "_stack"].pop()

        @property
        def tag_stack(self) -> List[torch.Tensor]:
            return self._get_brt_attr_stack("tag_stack")

        @tag_stack.setter
        def tag_stack(self, value):
            self._set_brt_attr_stack("tag_stack", value)

        @property
        def load_stack(self) -> List[int]:
            return self._get_brt_attr_stack("load_stack")

        @load_stack.setter
        def load_stack(self, value):
            self._set_brt_attr_stack("load_stack", value)

        @property
        def flow_initilized(self):
            return hasattr(self, "tag_stack") and hasattr(self, "load_stack")

        def flow_empty(self):
            if not self.tag_stack or not self.load_stack:
                return True
            return False

        @property
        def tag(self) -> torch.Tensor:
            return self._get_brt_attr("tag")

        @tag.setter
        def tag(self, value):
            self._set_brt_attr("tag", value)

        @property
        def load(self) -> int:
            return self._get_brt_attr("load")

        @load.setter
        def load(self, value):
            self._set_brt_attr("load", value)

        def pack(self, tag: torch.Tensor, load: int, **kwargs):
            self.init_flow()
            self._push_brt_attr("tag", tag)
            self._push_brt_attr("load", load)

            for attr in extra_attrs:
                value = kwargs.pop(attr, self.EXTRA_ATTRS_DEFAULT_VALUES[attr])
                self._push_brt_attr(attr, value)

            return self

        def unpack(self):
            assert self.flow_initilized and not self.flow_empty()
            tag = self._pop_brt_attr("tag")
            load = self._pop_brt_attr("load")
            extra_attrs_dict = {}

            for attr in extra_attrs:
                value = self._pop_brt_attr(attr)
                extra_attrs_dict[attr] = value

            return self, tag, load, extra_attrs_dict

        def deep_pack(
            self, tag_stack: List[torch.Tensor], load_stack: List[int], **kwargs
        ):
            assert isinstance(tag_stack, List) and isinstance(load_stack, List)
            self.tag_stack = tag_stack
            self.load_stack = load_stack

            for attr_stack in extra_attrs_stack:
                value = kwargs.pop(attr_stack, [])
                assert isinstance(value, List)
                self._set_brt_attr_stack(attr_stack, value)

            return self

        def deep_unpack(self):
            assert self.flow_initilized and not self.flow_empty()

            tag_stack = self.tag_stack
            load_stack = self.load_stack
            extra_attrs_stack_dict = {}

            for attr_stack in extra_attrs_stack:
                value = self._get_brt_attr_stack(attr_stack)
                extra_attrs_stack_dict[attr_stack] = value

            return self, tag_stack, load_stack, extra_attrs_stack_dict

        @property
        def pack_size(self) -> int:
            return len(self.tag_stack)

        def __repr__(self):
            return f"FlowTensor:\ndata: {super().__repr__()}\ntag_stack: {self.tag_stack}\nload stack: {self.load_stack}"

        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            if kwargs is None:
                kwargs = {}
            tag_stack, load_stack, extra_attrs_stack_dict = collect_attr_stack(args)
            assert tag_stack and load_stack
            ret = super().__torch_function__(func, types, args, kwargs)
            pack_ret(ret, tag_stack, load_stack, extra_attrs_stack_dict)
            return ret

    for attr in extra_attrs:
        setattr(FlowTensor, attr, flow_attr_factory(attr))
    for attr_stack in extra_attrs_stack:
        setattr(FlowTensor, attr_stack, flow_attr_stack_factory(attr_stack))

    return FlowTensor


FlowTensor = _make_flow_tensor([], [])


def make_flow_tensor(extra_attrs: List[str], default_values: List[Any]):
    global FlowTensor
    FlowTensor = _make_flow_tensor(extra_attrs, default_values)
    return FlowTensor


def init_flow_tensor(
    data: torch.Tensor, tag_stack: List[torch.Tensor], load_stack: List[int], **kwargs
) -> FlowTensor:
    data = data.as_subclass(FlowTensor)
    data.deep_pack(tag_stack, load_stack, **kwargs)
    return data


def deinit_flow_tensor(
    data: FlowTensor,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[int]]:
    data, tag_stack, load_stack, extra_attrs_stack_dict = data.deep_unpack()
    data = data.as_subclass(torch.Tensor)
    return data, tag_stack, load_stack, extra_attrs_stack_dict


# class FlowTensor(torch.Tensor):
#     CHECK_TAGS = False

#     def init_flow(self):
#         if not self.flow_initilized:
#             self.tags = []
#             self.loads = []

#     @property
#     def flow_initilized(self):
#         return hasattr(self, "tags") and hasattr(self, "loads")

#     @property
#     def tag(self) -> torch.Tensor:
#         if self.flow_empty():
#             print(self)
#             logger.error(
#                 "Trying to get tag from an FlowTensor without packed tag and load"
#             )
#             # return torch.zeros(0, 1, dtype=torch.int64, device=self.device)
#         return self.tags[-1]

#     @property
#     def load(self) -> int:
#         if self.flow_empty():
#             logger.warning(
#                 "Trying to get load from an FlowTensor without packed tag and load"
#             )
#             return 0
#         return self.loads[-1]

#     def pack(self, tag: torch.Tensor, load: int):
#         self.init_flow()
#         assert isinstance(tag, torch.Tensor), "tag must be a torch.Tensor for packing"
#         self.tags.append(tag)
#         self.loads.append(load)
#         return self

#     def unpack(self):
#         if not hasattr(self, "tags") or not hasattr(self, "loads"):
#             logger.error("Unpacking a not initialized FlowTensor")
#         tag = self.tags.pop()
#         load = self.loads.pop()
#         return self, tag, load

#     def deep_pack(self, tags: List[torch.Tensor], loads: List[int]):
#         self.tags = copy.deepcopy(tags)
#         self.loads = copy.deepcopy(loads)
#         return self

#     def deep_unpack(self):
#         if not hasattr(self, "tags") or not hasattr(self, "loads"):
#             logger.error("Unpacking a not initialized FlowTensor")
#         tags, self.tags = self.tags, []
#         loads, self.loads = self.loads, []
#         return self, tags, loads

#     @property
#     def pack_size(self) -> int:
#         return len(self.tags)

#     def flow_empty(self):
#         tags = getattr(self, "tags", [])
#         loads = getattr(self, "loads", [])
#         if not tags or not loads:
#             return True
#         return False

#     def __repr__(self):
#         return f"FlowTensor:\ndata: {super().__repr__()}\ncurrent tag: {self.tag}\ncurrent load: {self.load}"

#     @classmethod
#     def __torch_function__(cls, func, types, args=(), kwargs=None):
#         if kwargs is None:
#             kwargs = {}
#         all_tags, all_loads = collect_tags_loads(args)
#         assert len(all_tags[0]) > 0 and len(all_loads[0]) > 0
#         if cls.CHECK_TAGS:
#             for tags in all_tags:
#                 for i in range(len(tags)):
#                     assert torch.allclose(all_tags[0][i], tags[i])
#         ret = super().__torch_function__(func, types, args, kwargs)
#         pack_ret(ret, all_tags[0], all_loads[0])
#         return ret


# def collect_tags_loads(args):
#     all_tags = []
#     all_loads = []

#     if isinstance(args, FlowTensor):
#         all_tags.append(args.tags)
#         all_loads.append(args.loads)

#     if isinstance(args, (Tuple, List)):
#         for a in args:
#             _all_tags, _all_loads = collect_tags_loads(a)
#             all_tags.extend(_all_tags)
#             all_loads.extend(_all_loads)

#     return all_tags, all_loads


# def pack_ret(ret, tags, loads):

#     if isinstance(ret, FlowTensor):
#         ret = ret.deep_pack(tags, loads)

#     if isinstance(ret, (Tuple, List)):
#         ret = type(ret)(pack_ret(t, tags, loads) for t in ret)

#     return ret
