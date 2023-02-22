import copy
from typing import Any, Dict, List, Tuple, Union, Callable

import torch
from brt.runtime import log

__all__ = [
    "Cell",
    "init_cell",
    "deinit_cell",
    "to_cell",
    "to_tensor",
    "collect_cell_attr_stack",
    "pack_proto_attr_stack",
]

logger = log.get_logger(__file__)


def proto_attr_factory(name):
    def getter(self):
        return self._get_proto_attr(name)

    def setter(self, value):
        self._set_proto_attr(name, value)

    return property(getter, setter, doc=f"brt Mono attr: {name}")


def proto_attr_stack_factory(name):
    def getter(self):
        return self._get_proto_attr_stack(name)

    def setter(self, value):
        self._set_proto_attr_stack(name, value)

    return property(getter, setter, doc=f"brt Mono attr stack: {name}")


def collect_cell_attr_stack(
    args,
) -> Tuple[List[torch.Tensor], List[int], Dict[str, List[Any]]]:
    """collect all attr stack from args
        We have a big assumption here that the args of Mono type will shared the same attr stack.
        Common nn.Module only transmit them the attr stack of Mono without modification.
        Therefore, we only pass reference of the attr stack instead of a deep copy.
    Args:
        args (_type_): args of a torch function handled by __torch_function__

    Returns:
        all_attr_stack: including mandatory tag stack, load stack and extra attrs stack
    """
    all_tag_stack = None
    all_load_stack = None
    all_extra_attrs_stack_dict = {
        attr_stack: [] for attr_stack in Cell.EXTRA_ATTRS_STACK
    }

    if isinstance(args, Cell):
        all_tag_stack = args.tag_stack
        all_load_stack = args.load_stack
        for attr_stack in Cell.EXTRA_ATTRS_STACK:
            all_extra_attrs_stack_dict[attr_stack] = getattr(args, attr_stack)
        return all_tag_stack, all_load_stack, all_extra_attrs_stack_dict

    if isinstance(args, (Tuple, List)):
        for a in args:
            (
                _all_tags,
                _all_loads,
                _all_extra_attrs_stack_dict,
            ) = collect_cell_attr_stack(a)
            if _all_tags is not None and _all_loads is not None:
                return _all_tags, _all_loads, _all_extra_attrs_stack_dict

    return all_tag_stack, all_load_stack, all_extra_attrs_stack_dict


def pack_proto_attr_stack(
    ret,
    tag_stack: List[torch.Tensor],
    load_stack: List[int],
    extra_attrs_stack_dict: Dict[str, List[Any]] = None,
):

    if isinstance(ret, Cell):
        extra_attrs_stack_dict = extra_attrs_stack_dict or {}
        ret = ret.deep_pack(tag_stack, load_stack, **extra_attrs_stack_dict)

    if isinstance(ret, (Tuple, List)):
        ret = type(ret)(
            pack_proto_attr_stack(t, tag_stack, load_stack, extra_attrs_stack_dict)
            for t in ret
        )

    return ret


class Cell(torch.Tensor):
    SHALLOW_TRANSPORT = True
    CHECK_TAGS = False

    def init_cell(self, dims: List[int]):
        if not self.cell_initilized:
            self.__dict__["cell_tag_stack"] = []
            self.__dict__["cell_load_stack"] = []

    def annotation_empty(self):
        if not self.tag_stack or not self.load_stack:
            return True
        return False

    @property
    def tag_stack(self) -> List[torch.Tensor]:
        return self._get_attr_stack("tag_stack")

    @tag_stack.setter
    def tag_stack(self, value):
        self._set_attr_stack("tag_stack", value)

    @property
    def load_stack(self) -> List[int]:
        return self._get_attr_stack("load_stack")

    @load_stack.setter
    def load_stack(self, value):
        self._set_attr_stack("load_stack", value)

    @property
    def cell_initilized(self):
        return hasattr(self, "cell_tag_stack") and hasattr(self, "cell_load_stack")

    @property
    def tag(self) -> torch.Tensor:
        return self._get_attr("tag")

    @tag.setter
    def tag(self, value):
        self._set_attr("tag", value)

    @property
    def load(self) -> int:
        return self._get_attr("load")

    @load.setter
    def load(self, value):
        self._set_attr("load", value)

    def _get_attr_stack(self, attr_stack):
        return self.__dict__["cell_" + attr_stack]

    def _set_attr_stack(self, attr_stack, value):
        """We need at least a shadow copy here because the attr_stack can be shared with other Mono.
        otherwise, modifying other Mono will modify the attr_stack of this Mono.
        """
        if Cell.SHALLOW_TRANSPORT:
            self.__dict__["cell_" + attr_stack] = value
        else:
            self.__dict__["cell_" + attr_stack] = copy.copy(value)

    def _get_attr(self, attr):
        assert (
            not self.annotation_empty()
        ), f"Trying to get {attr} from a Mono without packed tag and load"
        return self.__dict__["cell_" + attr + "_stack"][-1]

    def _set_attr(self, attr, value):
        assert (
            not self.annotation_empty()
        ), f"Trying to set {attr} to {value} for a Mono without packed tag and load"
        self.__dict__["cell_" + attr + "_stack"][-1] = value

    def _push_attr(self, attr, value):
        self.__dict__["cell_" + attr + "_stack"].append(value)

    def _pop_attr(self, attr):
        return self.__dict__["cell_" + attr + "_stack"].pop()

    def pack(self, tag: torch.Tensor, load: int, **kwargs):
        self.init_cell()
        self._push_attr("tag", tag)
        self._push_attr("load", load)

        for attr in Cell.EXTRA_ATTRS:
            value = kwargs.pop(attr, Cell.EXTRA_ATTRS_DEFAULT_VALUES[attr])
            self._push_attr(attr, value)

        return self

    def unpack(self):
        assert self.cell_initilized and not self.annotation_empty()
        tag = self._pop_attr("tag")
        load = self._pop_attr("load")
        extra_attrs_dict = {}

        for attr in Cell.EXTRA_ATTRS:
            value = self._pop_attr(attr)
            extra_attrs_dict[attr] = value

        return self, tag, load, extra_attrs_dict

    def deep_pack(self, tag_stack: List[torch.Tensor], load_stack: List[int], **kwargs):
        assert isinstance(tag_stack, List) and isinstance(
            load_stack, List
        ), "tag_stack and load_stack must be list for deep_pack"
        self.tag_stack = tag_stack
        self.load_stack = load_stack

        for attr_stack in Cell.EXTRA_ATTRS_STACK:
            value = kwargs.pop(
                attr_stack, Cell.EXTRA_ATTRS_STACK_DEFAULT_VALUES[attr_stack]
            )
            assert isinstance(value, List)
            self._set_attr_stack(attr_stack, value)

        return self

    def deep_unpack(self):
        assert self.cell_initilized

        tag_stack, self.tag_stack = self.tag_stack, []
        load_stack, self.load_stack = self.load_stack, []
        extra_attrs_stack_dict = {}

        for attr_stack in Cell.EXTRA_ATTRS_STACK:
            value = self._get_attr_stack(attr_stack)
            extra_attrs_stack_dict[attr_stack] = value
            self._set_attr_stack(attr_stack, [])

        return self, tag_stack, load_stack, extra_attrs_stack_dict

    def copy_stacks(self):
        assert self.cell_initilized
        if Cell.SHALLOW_TRANSPORT:
            tag_stack = self.tag_stack
            load_stack = self.load_stack
            extra_attrs_stack_dict = {}

            for attr_stack in Cell.EXTRA_ATTRS_STACK:
                value = self._get_attr_stack(attr_stack)
                extra_attrs_stack_dict[attr_stack] = value
        else:
            tag_stack = copy.copy(self.tag_stack)
            load_stack = copy.copy(self.load_stack)
            extra_attrs_stack_dict = {}

            for attr_stack in Cell.EXTRA_ATTRS_STACK:
                value = self._get_attr_stack(attr_stack)
                extra_attrs_stack_dict[attr_stack] = copy.copy(value)

        return self, tag_stack, load_stack, extra_attrs_stack_dict

    @property
    def stack_size(self) -> int:
        return len(self.tag_stack)

    def __repr__(self):
        return f"{super().__repr__()}\ntag_stack: {self.tag_stack}\nload stack: {self.load_stack})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        make sure we don't call the built-in functions of torch.Tensor redefined in Mono
        inside the __torch_function__, otherwise it causes infinite recursion
        """
        if kwargs is None:
            kwargs = {}
        tag_stack, load_stack, extra_attrs_stack_dict = collect_cell_attr_stack(args)
        assert tag_stack is not None and load_stack is not None
        ret = super().__torch_function__(func, types, args, kwargs)
        pack_proto_attr_stack(ret, tag_stack, load_stack, extra_attrs_stack_dict)
        return ret


def init_cell(
    tensor: torch.Tensor,
    tag_stack: List[torch.Tensor] = None,
    load_stack: List[int] = None,
    extra_attrs_stack_dict: Dict[str, List[Any]] = None,
) -> Cell:
    cell = tensor.as_subclass(Cell)
    extra_attrs_stack_dict = extra_attrs_stack_dict or {}
    if tag_stack and load_stack:
        cell.deep_pack(tag_stack, load_stack, **extra_attrs_stack_dict)
    else:
        cell.init_cell()
    return cell


def deinit_cell(
    cell: Cell,
    retrieve_attr=True,
) -> Union[Tuple[torch.Tensor, List[torch.Tensor], List[int]], torch.Tensor]:
    if retrieve_attr:
        (
            cell,
            tag_stack,
            load_stack,
            extra_attrs_stack_dict,
        ) = cell.deep_unpack()
        tensor = cell.as_subclass(torch.Tensor)
        return tensor, tag_stack, load_stack, extra_attrs_stack_dict
    else:
        return cell.as_subclass(torch.Tensor)


def annotate(tensor: torch.Tensor, from_mono: Cell) -> Cell:
    return init_cell(tensor, *collect_cell_attr_stack(from_mono))


def unannotate(mono: Cell, return_att=False) -> torch.Tensor:
    return deinit_cell(mono)[0]


def to_cell(_torch_tensor: torch.Tensor):
    """
    restore a torch.Tensor to a Mono without any pack operation
    """
    proto_tensor: Cell = _torch_tensor.as_subclass(Cell)
    assert proto_tensor.cell_initilized
    return proto_tensor


def to_tensor(proto_tensor: Cell, return_stack=False):
    """
    we avoid broadcasting stack information by restore a Mono to
    torch.Tensor when we do not need it, e.g., inside the routers
    """
    if return_stack:
        (
            proto_tensor,
            tag_stack,
            load_stack,
            extra_attrs_stack_dict,
        ) = proto_tensor.copy_stacks()
    torch_tensor = proto_tensor.as_subclass(torch.Tensor)
    if return_stack:
        return torch_tensor, tag_stack, load_stack, extra_attrs_stack_dict
    else:
        return torch_tensor


def cell_tunnel(
    _func: Callable,
    *args,
    **kwargs,
) -> Union[Tuple[torch.Tensor, List[torch.Tensor], List[int]], torch.Tensor]:
    """
    A tunnel to convert the torch.Tensor to Mono and back to torch.Tensor
    """
    args = list(args)
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            args[i] = to_cell(arg)
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            kwargs[key] = to_cell(value)
    ret = _func(*args, **kwargs)
    if isinstance(ret, torch.Tensor):
        return to_tensor(ret)
    elif isinstance(ret, tuple):
        ret = list(ret)
        for i, arg in enumerate(ret):
            if isinstance(arg, torch.Tensor):
                ret[i] = to_tensor(arg)
        return tuple(ret)
