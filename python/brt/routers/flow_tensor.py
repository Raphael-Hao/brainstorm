import copy
from typing import Any, Dict, List, Tuple, TypeVar

import torch
from brt.common import log

__all__ = [
    "FlowTensor",
    "reset_flow_tensor_cls",
    "make_flow_tensor_cls",
    "init_flow_tensor",
    "deinit_flow_tensor",
    "to_flow_tensor",
    "to_torch_tensor",
]

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
    all_tag_stack = None
    all_load_stack = None
    all_extra_attrs_stack_dict = {
        attr_stack: [] for attr_stack in FlowTensor.EXTRA_ATTRS_STACK
    }

    if isinstance(args, FlowTensor):
        all_tag_stack = args.tag_stack
        all_load_stack = args.load_stack
        for attr_stack in FlowTensor.EXTRA_ATTRS_STACK:
            all_extra_attrs_stack_dict[attr_stack] = getattr(args, attr_stack)
        return all_tag_stack, all_load_stack, all_extra_attrs_stack_dict

    if isinstance(args, (Tuple, List)):
        for a in args:
            _all_tags, _all_loads, _all_extra_attrs_stack_dict = collect_attr_stack(a)
            if _all_tags is not None and _all_loads is not None:
                return _all_tags, _all_loads, _all_extra_attrs_stack_dict

    return all_tag_stack, all_load_stack, all_extra_attrs_stack_dict


def pack_ret(
    ret,
    tag_stack: List[torch.Tensor],
    load_stack: List[int],
    extra_attrs_stack_dict: Dict[str, List[Any]] = {},
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
    EXTRA_ATTRS = []
    EXTRA_ATTRS_STACK = []
    EXTRA_ATTRS_DEFAULT_VALUES = {}
    EXTRA_ATTRS_STACK_DEFAULT_VALUES = {}

    def init_flow(self):
        if not self.flow_initilized:
            self.__dict__["brt_tag_stack"] = []
            self.__dict__["brt_load_stack"] = []
            for attr_stack in FlowTensor.EXTRA_ATTRS_STACK:
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

    def flow_empty(self):
        if not self.tag_stack or not self.load_stack:
            return True
        return False

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
        return hasattr(self, "brt_tag_stack") and hasattr(self, "brt_load_stack")

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

        for attr in FlowTensor.EXTRA_ATTRS:
            print(attr)
            value = kwargs.pop(attr, FlowTensor.EXTRA_ATTRS_DEFAULT_VALUES[attr])
            self._push_brt_attr(attr, value)

        return self

    def unpack(self):
        assert self.flow_initilized and not self.flow_empty()
        tag = self._pop_brt_attr("tag")
        load = self._pop_brt_attr("load")
        extra_attrs_dict = {}

        for attr in FlowTensor.EXTRA_ATTRS:
            value = self._pop_brt_attr(attr)
            extra_attrs_dict[attr] = value

        return self, tag, load, extra_attrs_dict

    def deep_pack(self, tag_stack: List[torch.Tensor], load_stack: List[int], **kwargs):
        assert isinstance(tag_stack, List) and isinstance(
            load_stack, List
        ), "tag_stack and load_stack must be list for deep_pack"
        self.tag_stack = tag_stack
        self.load_stack = load_stack

        for attr_stack in FlowTensor.EXTRA_ATTRS_STACK:
            value = kwargs.pop(
                attr_stack, FlowTensor.EXTRA_ATTRS_STACK_DEFAULT_VALUES[attr_stack]
            )
            assert isinstance(value, List)
            self._set_brt_attr_stack(attr_stack, value)

        return self

    def deep_unpack(self):
        assert self.flow_initilized

        tag_stack, self.tag_stack = self.tag_stack, []
        load_stack, self.load_stack = self.load_stack, []
        extra_attrs_stack_dict = {}

        for attr_stack in FlowTensor.EXTRA_ATTRS_STACK:
            value = self._get_brt_attr_stack(attr_stack)
            extra_attrs_stack_dict[attr_stack] = value
            self._set_brt_attr_stack(attr_stack, [])

        return self, tag_stack, load_stack, extra_attrs_stack_dict

    def copy_stacks(self):
        assert self.flow_initilized

        tag_stack = copy.copy(self.tag_stack)
        load_stack = copy.copy(self.load_stack)
        extra_attrs_stack_dict = {}

        for attr_stack in FlowTensor.EXTRA_ATTRS_STACK:
            value = self._get_brt_attr_stack(attr_stack)
            extra_attrs_stack_dict[attr_stack] = copy.copy(value)

        return self, tag_stack, load_stack, extra_attrs_stack_dict

    @property
    def stack_size(self) -> int:
        return len(self.tag_stack)

    def __repr__(self):
        return f"FlowTensor(\ndata: {super().__repr__()}\ntag_stack: {self.tag_stack}\nload stack: {self.load_stack})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        make sure we don't call the built-in functions of torch.Tensor redefined in FlowTensor
        inside the __torch_function__, otherwise it causes infinite recursion
        """
        if kwargs is None:
            kwargs = {}
        tag_stack, load_stack, extra_attrs_stack_dict = collect_attr_stack(args)
        assert tag_stack is not None and load_stack is not None
        ret = super().__torch_function__(func, types, args, kwargs)
        pack_ret(ret, tag_stack, load_stack, extra_attrs_stack_dict)
        return ret


def reset_flow_tensor_cls():
    global FlowTensor
    for attr in FlowTensor.EXTRA_ATTRS:
        delattr(FlowTensor, attr)
    for attr_stack in FlowTensor.EXTRA_ATTRS_STACK:
        delattr(FlowTensor, attr_stack)
    FlowTensor.EXTRA_ATTRS = []
    FlowTensor.EXTRA_ATTRS_STACK = []
    FlowTensor.EXTRA_ATTRS_DEFAULT_VALUES = {}
    FlowTensor.EXTRA_ATTRS_STACK_DEFAULT_VALUES = {}
    return FlowTensor


def make_flow_tensor_cls(extra_attrs: List[str], default_values: List[Any]):

    global FlowTensor
    reset_flow_tensor_cls()

    # verify that the extra_attrs and default_values are the same length
    extra_attrs_set = set(extra_attrs)
    assert (len(extra_attrs_set) == len(extra_attrs)) and (
        len(extra_attrs_set) == len(default_values)
    )
    extra_attrs_stack = [attr + "_stack" for attr in extra_attrs]

    # modify the FlowTensor class extra attrs names and default values
    FlowTensor.EXTRA_ATTRS = extra_attrs
    FlowTensor.EXTRA_ATTRS_STACK = extra_attrs_stack
    for i in range(len(extra_attrs)):
        attr = extra_attrs[i]
        FlowTensor.EXTRA_ATTRS_DEFAULT_VALUES[attr] = default_values[i]
        attr_stack = extra_attrs_stack[i]
        FlowTensor.EXTRA_ATTRS_STACK_DEFAULT_VALUES[attr_stack] = [default_values[i]]

    # register property setter and getter for the extra attrs stack
    for attr_stack in FlowTensor.EXTRA_ATTRS_STACK:
        setattr(FlowTensor, attr_stack, flow_attr_stack_factory(attr_stack))

    # register property setter and getter for the extra attrs
    for attr in FlowTensor.EXTRA_ATTRS:
        setattr(FlowTensor, attr, flow_attr_factory(attr))

    return FlowTensor


def init_flow_tensor(
    _torch_tensor: torch.Tensor,
    tag_stack: List[torch.Tensor] = [],
    load_stack: List[int] = [],
    **kwargs,
) -> FlowTensor:
    _flow_tensor: FlowTensor = _torch_tensor.as_subclass(FlowTensor)
    if tag_stack and load_stack:
        _flow_tensor.deep_pack(tag_stack, load_stack, **kwargs)
    else:
        _flow_tensor.init_flow()
    return _flow_tensor


def deinit_flow_tensor(
    _flow_tensor: FlowTensor,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[int]]:
    (
        _flow_tensor,
        tag_stack,
        load_stack,
        extra_attrs_stack_dict,
    ) = _flow_tensor.deep_unpack()
    _torch_tensor = _flow_tensor.as_subclass(torch.Tensor)
    return _torch_tensor, tag_stack, load_stack, extra_attrs_stack_dict


def to_flow_tensor(_torch_tensor: torch.Tensor):
    """
    restore a torch.Tensor to a FlowTensor without any pack operation
    """
    _flow_tensor: FlowTensor = _torch_tensor.as_subclass(FlowTensor)
    assert _flow_tensor.flow_initilized
    return _flow_tensor


def to_torch_tensor(_flow_tensor: FlowTensor, copy_stack=False):
    """
    we avoid broadcasting stack information by restore a FlowTensor to
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
