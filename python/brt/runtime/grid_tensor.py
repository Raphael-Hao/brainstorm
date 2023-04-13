import copy
from typing import Any, Dict, List, Tuple, Union, Callable

import torch
import torch.nn as nn
from brt.trace.leaf_node import register_leaf_node
from brt.runtime import log

__all__ = [
    "GridTensor",
    "init_grid_tensor",
    "deinit_grid_tensor",
    "to_grid_tensor",
    "to_torch_tensor",
    "collect_cell_attr",
    "attach_cell_attr",
]

logger = log.get_logger(__file__)


def collect_cell_attr(
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
    all_extra_attr_dict = None

    if isinstance(args, GridTensor):
        all_tag_stack = args.tag_stack
        all_load_stack = args.load_stack
        all_extra_attr_dict = args.extra_attr_dict
        return (
            all_tag_stack,
            all_load_stack,
            all_extra_attr_dict,
        )

    if isinstance(args, (Tuple, List)):
        for a in args:
            (_all_tags, _all_loads, _all_extra_attr_dict,) = collect_cell_attr(a)
            if _all_tags is not None and _all_loads is not None:
                return (
                    _all_tags,
                    _all_loads,
                    _all_extra_attr_dict,
                )

    return (
        all_tag_stack,
        all_load_stack,
        all_extra_attr_dict,
    )


def attach_cell_attr(
    ret,
    tag_stack: List[torch.Tensor],
    load_stack: List[int],
    extra_attrs_stack_dict: Dict[str, List[Any]] = None,
):

    if isinstance(ret, GridTensor):
        extra_attrs_stack_dict = extra_attrs_stack_dict or {}
        ret = ret.deep_pack(tag_stack, load_stack, **extra_attrs_stack_dict)

    if isinstance(ret, (Tuple, List)):
        ret = type(ret)(
            attach_cell_attr(t, tag_stack, load_stack, extra_attrs_stack_dict)
            for t in ret
        )

    return ret


class GridTensor(torch.Tensor):
    SHALLOW_TRANSPORT = True
    CHECK_TAGS = False

    def init_cell(self):
        if not self.cell_initilized:
            self.__dict__["cell_tag_stack"] = []
            self.__dict__["cell_load_stack"] = []
            self.__dict__["cell_extra_attr_dict"] = {}

    def annotation_empty(self):
        if not self.tag_stack or not self.load_stack:
            return True
        return False

    @property
    def cell_initilized(self):
        return (
            hasattr(self, "cell_tag_stack")
            and hasattr(self, "cell_load_stack")
            and hasattr(self, "cell_extra_attr_dict")
        )

    @property
    def tag(self) -> torch.Tensor:
        return self._get_stacked_attr("tag")

    @tag.setter
    def tag(self, value):
        self._set_stacked_attr("tag", value)

    @property
    def load(self) -> int:
        return self._get_stacked_attr("load")

    @load.setter
    def load(self, value):
        self._set_stacked_attr("load", value)

    def set_extra_attr(self, key, value):
        assert key in self.__dict__["cell_extra_attr_dict"]
        if GridTensor.SHALLOW_TRANSPORT:
            self.__dict__["cell_extra_attr_dict"][key] = value
        else:
            self.__dict__["cell_extra_attr_dict"][key] = copy.copy(value)

    def get_extra_attr(self, key):
        assert key in self.__dict__["cell_extra_attr_dict"]
        return self.__dict__["cell_extra_attr_dict"][key]

    @property
    def tag_stack(self) -> List[torch.Tensor]:
        return self._get_cell_attr("tag_stack")

    @tag_stack.setter
    def tag_stack(self, value):
        self._set_cell_attr("tag_stack", value)

    @property
    def load_stack(self) -> List[torch.Tensor]:
        return self._get_cell_attr("load_stack")

    @load_stack.setter
    def load_stack(self, value):
        self._set_cell_attr("load_stack", value)

    @property
    def extra_attr_dict(self) -> Dict[str, Any]:
        return self._get_cell_attr("extra_attr_dict")

    @extra_attr_dict.setter
    def extra_attr_dict(self, value):
        self._set_cell_attr("extra_attr_dict", value)

    def _get_cell_attr(self, cell_attr):
        assert cell_attr in ["tag_stack", "load_stack", "extra_attr_dict"]
        return self.__dict__["cell_" + cell_attr]

    def _set_cell_attr(self, cell_attr, value):
        """We need at least a shadow copy here because the attr_stack can be shared with other GridTensor.
        otherwise, modifying other GridTensor will modify the attr_stack of this GridTensor.
        """
        if GridTensor.SHALLOW_TRANSPORT:
            self.__dict__["cell_" + cell_attr] = value
        else:
            self.__dict__["cell_" + cell_attr] = copy.copy(value)

    def _get_stacked_attr(self, attr):
        return self.__dict__["cell_" + attr + "_stack"][-1]

    def _set_stacked_attr(self, attr, value):
        self.__dict__["cell_" + attr + "_stack"][-1] = value

    def _push_stacked_attr(self, attr, value):
        self.__dict__["cell_" + attr + "_stack"].append(value)

    def _pop_stacked_attr(self, attr):
        return self.__dict__["cell_" + attr + "_stack"].pop()

    def _push_dicted_extra_attr(self, attr, value):
        assert attr not in self.__dict__["cell_extra_attr_dict"]
        self.__dict__["cell_extra_attr_dict"][attr] = value

    def _pop_dicted_extra_attr(self, attr):
        assert attr in self.__dict__["cell_extra_attr_dict"]
        return self.__dict__["cell_extra_attr_dict"].pop(attr)

    def pack(self, tag: torch.Tensor, load: torch.Tensor, **kwargs):
        self.init_cell()
        self._push_stacked_attr("tag", tag)
        self._push_stacked_attr("load", load)
        for k, v in kwargs.items():
            self._push_dicted_extra_attr(k, v)

        return self

    def unpack(self, *args):
        assert self.cell_initilized and not self.annotation_empty()
        tag = self._pop_stacked_attr("tag")
        load = self._pop_stacked_attr("load")
        extra_attr_dict = {}
        for k in args:
            extra_attr_dict[k] = self._pop_dicted_extra_attr(k)

        return self, tag, load, extra_attr_dict

    def deep_pack(self, tag_stack: List[torch.Tensor], load_stack: List[int], **kwargs):
        assert isinstance(tag_stack, List) and isinstance(
            load_stack, List
        ), "tag_stack and load_stack must be list for deep_pack"
        self.tag_stack = tag_stack
        self.load_stack = load_stack
        self.extra_attr_dict = {}
        for k, v in kwargs.items():
            self._push_dicted_extra_attr(k, v)

        return self

    def deep_unpack(self):
        assert self.cell_initilized

        tag_stack, self.tag_stack = self.tag_stack, []
        load_stack, self.load_stack = self.load_stack, []
        extra_attr_dict, self.extra_attr_dict = self.extra_attr_dict, {}

        return self, tag_stack, load_stack, extra_attr_dict

    def copy_cell_attr(self):
        assert self.cell_initilized
        if GridTensor.SHALLOW_TRANSPORT:
            tag_stack = self.tag_stack
            load_stack = self.load_stack
            extra_attr_dict = self.extra_attr_dict
        else:
            tag_stack = copy.copy(self.tag_stack)
            load_stack = copy.copy(self.load_stack)
            extra_attr_dict = copy.copy(self.extra_attr_dict)

        return self, tag_stack, load_stack, extra_attr_dict

    @property
    def stack_size(self) -> int:
        return len(self.tag_stack)

    def __repr__(self):
        return f"{super().__repr__()}\ntag_stack: {self.tag_stack}\nload stack: {self.load_stack}\nextra_attr_dict: {self.extra_attr_dict}"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        make sure we don't call the built-in functions of torch.Tensor redefined in Mono
        inside the __torch_function__, otherwise it causes infinite recursion
        """
        if kwargs is None:
            kwargs = {}
        tag_stack, load_stack, extra_attr_dict = collect_cell_attr(args)
        # assert tag_stack is not None and load_stack is not None
        ret = super().__torch_function__(func, types, args, kwargs)
        attach_cell_attr(ret, tag_stack, load_stack, extra_attr_dict)
        return ret


def init_grid_tensor(
    tensor: torch.Tensor,
    tag_stack: List[torch.Tensor] = None,
    load_stack: List[torch.Tensor] = None,
    extra_attr_dict: Dict[str, List[Any]] = None,
) -> GridTensor:
    cell: GridTensor = tensor.as_subclass(GridTensor)
    extra_attr_dict = extra_attr_dict or {}
    if tag_stack and load_stack:
        cell.deep_pack(tag_stack, load_stack, **extra_attr_dict)
    else:
        cell.init_cell()
    return cell


def deinit_grid_tensor(
    grid_t: GridTensor, retrieve_attr=True,
) -> Union[
    Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], Dict[str, Any]],
    torch.Tensor,
]:
    if retrieve_attr:
        (grid_t, tag_stack, load_stack, extra_attrs_stack_dict,) = grid_t.deep_unpack()
        t = grid_t.as_subclass(torch.Tensor)
        return t, tag_stack, load_stack, extra_attrs_stack_dict
    else:
        return grid_t.as_subclass(torch.Tensor)


def init_grid_tensor_from(tensor: torch.Tensor, from_gird_tensor: GridTensor):
    _t, tag_stack, load_stack, extra_attr_dict = deinit_grid_tensor(
        from_gird_tensor, True
    )
    return init_grid_tensor(tensor, tag_stack, load_stack, extra_attr_dict)


def to_grid_tensor(torch_t: torch.Tensor, tag_stack=None, load_stack=None, extra_attr_dict=None):
    """
    restore a torch.Tensor to a Mono without any pack operation
    """
    grid_t = init_grid_tensor(torch_t, tag_stack, load_stack, extra_attr_dict)
    return grid_t


def to_torch_tensor(grid_t: GridTensor, retrieve_attr=False):
    """
    we avoid broadcasting stack information by restore a GridTensor to
    torch.Tensor when we do not need it, e.g., inside the routers
    """
    if retrieve_attr:
        (
            grid_t,
            tag_stack,
            load_stack,
            extra_attrs_dict,
        ) = grid_t.copy_cell_attr()
    torch_tensor = grid_t.as_subclass(torch.Tensor)
    if retrieve_attr:
        return torch_tensor, tag_stack, load_stack, extra_attrs_dict
    else:
        return torch_tensor



@register_leaf_node
class Annotator(nn.Module):
    def __init__(self, dims: List[int], cell_shape: List[int]=None, gridding: bool=True):
        super().__init__()
        assert isinstance(dims, list)
        dims = sorted(dims)
        self.dims = dims
        if cell_shape is not None:
            assert isinstance(cell_shape, list)
        self.cell_shape = cell_shape
        self.gridding = gridding

    def forward(self, t: torch.Tensor):
        assert self.dims[-1] < len(t.shape)
        if self.cell_shape is None:
            cell_shape = [t.shape[i] for i in range(len(t.shape)) if i not in self.dims]
        else:
            cell_shape = self.cell_shape
        transposed_dims = self.dims + [i for i in range(len(t.shape)) if i not in self.dims]
        transposed_tensor = t.permute(*transposed_dims).contiguous()
        reshaped_tensor = transposed_tensor.reshape(-1, *cell_shape)
        if self.gridding:
            if isinstance(reshaped_tensor, GridTensor):
                initialized_tensor = reshaped_tensor.pack(None, None)
            else:
                initialized_tensor = init_grid_tensor(reshaped_tensor, [None], [None])
        else:
            initialized_tensor = reshaped_tensor
        return initialized_tensor

def annotate(
    t: torch.Tensor, dims: List[int]=None, cell_shape: List[int] = None
) -> GridTensor:
    """Annotate a tensor with cell granularity

    Args:
        t (torch.Tensor): _description_
        dims (List[int]): _description_
        cell_shape (List[int], optional): _description_. Defaults to None.

    Returns:
        GridTensor: _description_
    """
    assert isinstance(dims, list)
    dims = sorted(dims)
    assert dims[-1] < len(t.shape)
    if cell_shape is not None:
        assert isinstance(cell_shape, list)
    else:
        cell_shape = [t.shape[i] for i in range(len(t.shape)) if i not in dims]
    transposed_dims = dims + [i for i in range(len(t.shape)) if i not in dims]
    transposed_tensor = t.permute(*transposed_dims).contiguous()
    reshaped_tensor = transposed_tensor.reshape(-1, *cell_shape)

    if isinstance(reshaped_tensor, GridTensor):
        initialized_tensor = reshaped_tensor.pack(None, None)
    else:
        initialized_tensor = init_grid_tensor(reshaped_tensor, [None], [None])

    return initialized_tensor