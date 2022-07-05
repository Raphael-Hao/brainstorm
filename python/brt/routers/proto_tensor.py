import copy
from typing import Any, Dict, List, Tuple

import torch
from brt.common import log

__all__ = [
    "ProtoTensor",
    "reset_proto_tensor_cls",
    "make_proto_tensor_cls",
    "init_proto_tensor",
    "deinit_proto_tensor",
    "to_proto_tensor",
    "to_torch_tensor",
    "collect_proto_attr_stack",
    "pack_proto_attr_stack",
]

logger = log.get_logger(__file__)


def proto_attr_factory(name):
    def getter(self):
        return self._get_proto_attr(name)

    def setter(self, value):
        self._set_proto_attr(name, value)

    return property(getter, setter, doc=f"brt ProtoTensor attr: {name}")


def proto_attr_stack_factory(name):
    def getter(self):
        return self._get_proto_attr_stack(name)

    def setter(self, value):
        self._set_proto_attr_stack(name, value)

    return property(getter, setter, doc=f"brt ProtoTensor attr stack: {name}")


def collect_proto_attr_stack(
    args,
) -> Tuple[List[torch.Tensor], List[int], Dict[str, List[Any]]]:
    """collect all attr stack from args
        We have a big assumption here that the args of ProtoTensor type will shared the same attr stack.
        Common nn.Module only transmit them the attr stack of ProtoTensor without modification.
    Args:
        args (_type_): args of a torch function handled by __torch_function__

    Returns:
        all_attr_stack: including mandatory tag stack, load stack and extra attrs stack
    """
    all_tag_stack = None
    all_load_stack = None
    all_extra_attrs_stack_dict = {
        attr_stack: [] for attr_stack in ProtoTensor.EXTRA_ATTRS_STACK
    }

    if isinstance(args, ProtoTensor):
        all_tag_stack = args.tag_stack
        all_load_stack = args.load_stack
        for attr_stack in ProtoTensor.EXTRA_ATTRS_STACK:
            all_extra_attrs_stack_dict[attr_stack] = getattr(args, attr_stack)
        return all_tag_stack, all_load_stack, all_extra_attrs_stack_dict

    if isinstance(args, (Tuple, List)):
        for a in args:
            (
                _all_tags,
                _all_loads,
                _all_extra_attrs_stack_dict,
            ) = collect_proto_attr_stack(a)
            if _all_tags is not None and _all_loads is not None:
                return _all_tags, _all_loads, _all_extra_attrs_stack_dict

    return all_tag_stack, all_load_stack, all_extra_attrs_stack_dict


def pack_proto_attr_stack(
    ret,
    tag_stack: List[torch.Tensor],
    load_stack: List[int],
    extra_attrs_stack_dict: Dict[str, List[Any]] = {},
):

    if isinstance(ret, ProtoTensor):
        ret = ret.deep_pack(tag_stack, load_stack, **extra_attrs_stack_dict)

    if isinstance(ret, (Tuple, List)):
        ret = type(ret)(
            pack_proto_attr_stack(t, tag_stack, load_stack, extra_attrs_stack_dict)
            for t in ret
        )

    return ret


class ProtoTensor(torch.Tensor):
    CHECK_TAGS = False
    EXTRA_ATTRS = []
    EXTRA_ATTRS_STACK = []
    EXTRA_ATTRS_DEFAULT_VALUES = {}
    EXTRA_ATTRS_STACK_DEFAULT_VALUES = {}

    def init_proto(self):
        if not self.proto_initilized:
            self.__dict__["proto_tag_stack"] = []
            self.__dict__["proto_load_stack"] = []
            for attr_stack in ProtoTensor.EXTRA_ATTRS_STACK:
                self.__dict__["proto_" + attr_stack] = []

    def _get_proto_attr_stack(self, attr_stack):
        return self.__dict__["proto_" + attr_stack]

    def _set_proto_attr_stack(self, attr_stack, value):
        """We need at least a shadow copy here because the attr_stack can be shared with other ProtoTensor.
        otherwise, modifying other ProtoTensor will modify the attr_stack of this ProtoTensor.
        """
        self.__dict__["proto_" + attr_stack] = copy.copy(value)

    def _get_proto_attr(self, attr):
        assert (
            not self.proto_empty()
        ), f"Trying to get {attr} from a ProtoTensor without packed tag and load"
        return self.__dict__["proto_" + attr + "_stack"][-1]

    def _set_proto_attr(self, attr, value):
        assert (
            not self.proto_empty()
        ), f"Trying to set {attr} to {value} for a ProtoTensor without packed tag and load"
        self.__dict__["proto_" + attr + "_stack"][-1] = value

    def _push_proto_attr(self, attr, value):
        self.__dict__["proto_" + attr + "_stack"].append(value)

    def _pop_proto_attr(self, attr):
        return self.__dict__["proto_" + attr + "_stack"].pop()

    def proto_empty(self):
        if not self.tag_stack or not self.load_stack:
            return True
        return False

    @property
    def tag_stack(self) -> List[torch.Tensor]:
        return self._get_proto_attr_stack("tag_stack")

    @tag_stack.setter
    def tag_stack(self, value):
        self._set_proto_attr_stack("tag_stack", value)

    @property
    def load_stack(self) -> List[int]:
        return self._get_proto_attr_stack("load_stack")

    @load_stack.setter
    def load_stack(self, value):
        self._set_proto_attr_stack("load_stack", value)

    @property
    def proto_initilized(self):
        return hasattr(self, "proto_tag_stack") and hasattr(self, "proto_load_stack")

    @property
    def tag(self) -> torch.Tensor:
        return self._get_proto_attr("tag")

    @tag.setter
    def tag(self, value):
        self._set_proto_attr("tag", value)

    @property
    def load(self) -> int:
        return self._get_proto_attr("load")

    @load.setter
    def load(self, value):
        self._set_proto_attr("load", value)

    def pack(self, tag: torch.Tensor, load: int, **kwargs):
        self.init_proto()
        self._push_proto_attr("tag", tag)
        self._push_proto_attr("load", load)

        for attr in ProtoTensor.EXTRA_ATTRS:
            value = kwargs.pop(attr, ProtoTensor.EXTRA_ATTRS_DEFAULT_VALUES[attr])
            self._push_proto_attr(attr, value)

        return self

    def unpack(self):
        assert self.proto_initilized and not self.proto_empty()
        tag = self._pop_proto_attr("tag")
        load = self._pop_proto_attr("load")
        extra_attrs_dict = {}

        for attr in ProtoTensor.EXTRA_ATTRS:
            value = self._pop_proto_attr(attr)
            extra_attrs_dict[attr] = value

        return self, tag, load, extra_attrs_dict

    def deep_pack(self, tag_stack: List[torch.Tensor], load_stack: List[int], **kwargs):
        assert isinstance(tag_stack, List) and isinstance(
            load_stack, List
        ), "tag_stack and load_stack must be list for deep_pack"
        self.tag_stack = tag_stack
        self.load_stack = load_stack

        for attr_stack in ProtoTensor.EXTRA_ATTRS_STACK:
            value = kwargs.pop(
                attr_stack, ProtoTensor.EXTRA_ATTRS_STACK_DEFAULT_VALUES[attr_stack]
            )
            assert isinstance(value, List)
            self._set_proto_attr_stack(attr_stack, value)

        return self

    def deep_unpack(self):
        assert self.proto_initilized

        tag_stack, self.tag_stack = self.tag_stack, []
        load_stack, self.load_stack = self.load_stack, []
        extra_attrs_stack_dict = {}

        for attr_stack in ProtoTensor.EXTRA_ATTRS_STACK:
            value = self._get_proto_attr_stack(attr_stack)
            extra_attrs_stack_dict[attr_stack] = value
            self._set_proto_attr_stack(attr_stack, [])

        return self, tag_stack, load_stack, extra_attrs_stack_dict

    def copy_stacks(self):
        assert self.proto_initilized

        tag_stack = copy.copy(self.tag_stack)
        load_stack = copy.copy(self.load_stack)
        extra_attrs_stack_dict = {}

        for attr_stack in ProtoTensor.EXTRA_ATTRS_STACK:
            value = self._get_proto_attr_stack(attr_stack)
            extra_attrs_stack_dict[attr_stack] = copy.copy(value)

        return self, tag_stack, load_stack, extra_attrs_stack_dict

    @property
    def stack_size(self) -> int:
        return len(self.tag_stack)

    def __repr__(self):
        return f"ProtoTensor(\ndata: {super().__repr__()}\ntag_stack: {self.tag_stack}\nload stack: {self.load_stack})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        make sure we don't call the built-in functions of torch.Tensor redefined in ProtoTensor
        inside the __torch_function__, otherwise it causes infinite recursion
        """
        if kwargs is None:
            kwargs = {}
        tag_stack, load_stack, extra_attrs_stack_dict = collect_proto_attr_stack(args)
        assert tag_stack is not None and load_stack is not None
        ret = super().__torch_function__(func, types, args, kwargs)
        pack_proto_attr_stack(ret, tag_stack, load_stack, extra_attrs_stack_dict)
        return ret


def reset_proto_tensor_cls():
    global ProtoTensor
    for attr in ProtoTensor.EXTRA_ATTRS:
        delattr(ProtoTensor, attr)
    for attr_stack in ProtoTensor.EXTRA_ATTRS_STACK:
        delattr(ProtoTensor, attr_stack)
    ProtoTensor.EXTRA_ATTRS = []
    ProtoTensor.EXTRA_ATTRS_STACK = []
    ProtoTensor.EXTRA_ATTRS_DEFAULT_VALUES = {}
    ProtoTensor.EXTRA_ATTRS_STACK_DEFAULT_VALUES = {}
    return ProtoTensor


def make_proto_tensor_cls(
    extra_attrs: List[str], default_values: List[Any], mode="add"
):

    global ProtoTensor
    if mode == "new":
        reset_proto_tensor_cls()

        # verify that the extra_attrs and default_values are the same length
        extra_attrs_set = set(extra_attrs)
        assert (len(extra_attrs_set) == len(extra_attrs)) and (
            len(extra_attrs_set) == len(default_values)
        )
        extra_attrs_stack = [attr + "_stack" for attr in extra_attrs]

        # modify the ProtoTensor class extra attrs names and default values
        ProtoTensor.EXTRA_ATTRS = extra_attrs
        ProtoTensor.EXTRA_ATTRS_STACK = extra_attrs_stack
        for i in range(len(extra_attrs)):
            attr = extra_attrs[i]
            ProtoTensor.EXTRA_ATTRS_DEFAULT_VALUES[attr] = default_values[i]
            attr_stack = extra_attrs_stack[i]
            ProtoTensor.EXTRA_ATTRS_STACK_DEFAULT_VALUES[attr_stack] = [
                default_values[i]
            ]

        # register property setter and getter for the extra attrs stack
        for attr_stack in ProtoTensor.EXTRA_ATTRS_STACK:
            setattr(ProtoTensor, attr_stack, proto_attr_stack_factory(attr_stack))

        # register property setter and getter for the extra attrs
        for attr in ProtoTensor.EXTRA_ATTRS:
            setattr(ProtoTensor, attr, proto_attr_factory(attr))
    elif mode == "add":
        extra_attrs_set = set(extra_attrs)
        assert (len(extra_attrs_set) == len(extra_attrs)) and (
            len(extra_attrs_set) == len(default_values)
        )
        extra_attrs_stack = [attr + "_stack" for attr in extra_attrs]

        # modify the ProtoTensor class extra attrs names and default values
        ProtoTensor.EXTRA_ATTRS.extend(extra_attrs)
        ProtoTensor.EXTRA_ATTRS_STACK.extend(extra_attrs_stack)
        for i in range(len(extra_attrs)):
            attr = extra_attrs[i]
            ProtoTensor.EXTRA_ATTRS_DEFAULT_VALUES[attr] = default_values[i]
            attr_stack = extra_attrs_stack[i]
            ProtoTensor.EXTRA_ATTRS_STACK_DEFAULT_VALUES[attr_stack] = [
                default_values[i]
            ]

        # register property setter and getter for the extra attrs stack
        for attr_stack in ProtoTensor.EXTRA_ATTRS_STACK:
            setattr(ProtoTensor, attr_stack, proto_attr_stack_factory(attr_stack))

        # register property setter and getter for the extra attrs
        for attr in ProtoTensor.EXTRA_ATTRS:
            setattr(ProtoTensor, attr, proto_attr_factory(attr))

    return ProtoTensor


def init_proto_tensor(
    _torch_tensor: torch.Tensor,
    tag_stack: List[torch.Tensor] = [None],
    load_stack: List[int] = [None],
    extra_attrs_stack_dict: Dict[str, List[Any]] = {},
) -> ProtoTensor:
    _flow_tensor: ProtoTensor = _torch_tensor.as_subclass(ProtoTensor)
    if tag_stack and load_stack:
        _flow_tensor.deep_pack(tag_stack, load_stack, **extra_attrs_stack_dict)
    else:
        _flow_tensor.init_proto()
    return _flow_tensor


def deinit_proto_tensor(
    _flow_tensor: ProtoTensor,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[int]]:
    (
        _flow_tensor,
        tag_stack,
        load_stack,
        extra_attrs_stack_dict,
    ) = _flow_tensor.deep_unpack()
    _torch_tensor = _flow_tensor.as_subclass(torch.Tensor)
    return _torch_tensor, tag_stack, load_stack, extra_attrs_stack_dict


def to_proto_tensor(_torch_tensor: torch.Tensor):
    """
    restore a torch.Tensor to a ProtoTensor without any pack operation
    """
    _flow_tensor: ProtoTensor = _torch_tensor.as_subclass(ProtoTensor)
    assert _flow_tensor.proto_initilized
    return _flow_tensor


def to_torch_tensor(_flow_tensor: ProtoTensor, copy_stack=False):
    """
    we avoid broadcasting stack information by restore a ProtoTensor to
    torch.Tensor when we do not need it, e.g., inside the routers
    """
    if copy_stack:
        (
            _flow_tensor,
            tag_stack,
            load_stack,
            extra_attrs_stack_dict,
        ) = _flow_tensor.copy_stacks()
    _torch_tensor = _flow_tensor.as_subclass(torch.Tensor)
    if copy_stack:
        return _torch_tensor, tag_stack, load_stack, extra_attrs_stack_dict
    else:
        return _torch_tensor
