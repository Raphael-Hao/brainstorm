# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import collections
import functools
import inspect
import numbers
import types
from io import IOBase
from typing import TypeVar, Union

import cloudpickle
import torch
from brt.common import log

from .base import (
    SerializableObject,
    Traceable,
    _argument_processor,
    _copy_class_wrapper_attributes,
    _formulate_arguments,
    _is_function,
    _make_class_traceable,
    _pickling_object,
    _unwrap_metaclass,
    inject_trace_info,
    torchscript_patch,
)
from .helper import check_wrapped, is_wrapped_with_trace

logger = log.get_logger(__file__)

__all__ = ["netlet"]
T = TypeVar("T")


def netlet(cls: T, netlet_tag: bool = True) -> T:
    """
    Decorator for annotating an nn.Module as a Netlet.
    TODO check why we don't need to switch forward method
    """
    if check_wrapped(cls, "netlet"):
        return cls

    cls = trace_netlet(cls)

    cls._brt_netlet = netlet_tag
    torchscript_patch(cls)
    return cls


def trace_netlet(
    cls_or_func: T = None, *, kw_only: bool = True, inheritable: bool = False
) -> Union[T, Traceable]:
    def wrap(cls_or_func):
        # already annotated, do nothing
        if is_wrapped_with_trace(cls_or_func):
            return cls_or_func
        if isinstance(cls_or_func, type):
            cls_or_func = _trace_netlet_cls(
                cls_or_func, kw_only, inheritable=inheritable
            )
        elif _is_function(cls_or_func):
            # cls_or_func = _trace_netlet_func(cls_or_func, kw_only)
            raise NotImplementedError(
                "Tracing function is not implemented for netlet yet."
            )
        else:
            raise TypeError(
                f"{cls_or_func} of type {type(cls_or_func)} is not supported to be traced. "
                "File an issue at https://github.com/microsoft/nni/issues if you believe this is a mistake."
            )
        cls_or_func._traced = True
        return cls_or_func

    # if we're being called as @trace()
    if cls_or_func is None:
        return wrap

    # if we are called without parentheses
    return wrap(cls_or_func)


def _trace_netlet_cls(base: T, kw_only, call_super=True, inheritable=False):
    # the implementation to trace a class is to store a copy of init arguments
    # this won't support class that defines a customized new but should work for most cases

    # This is trying to solve the case where superclass and subclass are both decorated with @brt.trace.
    # We use a metaclass to "unwrap" the superclass.
    # However, this doesn't work if:
    # 1. Base class already has a customized metaclass. We will raise error in that class.
    # 2. SerializableObject in ancester (instead of parent). I think this case is rare and I didn't handle this case yet. FIXME
    if type(base) is type and not inheritable:
        metaclass = _unwrap_metaclass
    else:
        metaclass = type
        if SerializableObject in inspect.getmro(base):
            raise TypeError(
                f"{base} has a superclass already decorated with trace, and it's using a customized metaclass {type(base)}. "
                "Please either use the default metaclass, or remove trace from the super-class."
            )

    class wrapper(SerializableObject, base, metaclass=metaclass):
        def __init__(self, *args, **kwargs):
            # store a copy of initial parameters
            args, kwargs = _formulate_arguments(
                base.__init__, args, kwargs, kw_only, is_class_init=True
            )

            # calling serializable object init to initialize the full object
            super().__init__(
                symbol=base, args=args, kwargs=kwargs, call_super=call_super
            )
            self.pt_forward = super().forward
            self.forward = self.brt_forward
            self.using_brt_forward = True

        @torch.jit.ignore
        def brt_forward(self, *inputs):
            logger.debug("using brt_forward")
            assert isinstance(
                inputs[0], torch.Tensor
            ), "BRT requires the first argument to be a tensor"
            if inputs[0].numel() == 0:
                return torch.zeros_like(inputs[0])
            return self.pt_forward(*inputs)

        @property
        def _brt_retn(self) -> int:
            # pt_fwd_sig = inspect.signature(self.pt_forward)
            return 1

        def __reduce__(self):
            # The issue that decorator and pickler doesn't play well together is well known.
            # The workaround solution is to use a fool class (_pickling_object) which pretends to be the pickled object.
            # We then put the original type, as well as args and kwargs in its `__new__` argument.
            # I suspect that their could still be problems when things get complex,
            # e.g., the wrapped class has a custom pickling (`__reduce__``) or `__new__`.
            # But it can't be worse because the previous pickle doesn't work at all.
            #
            # Linked issue: https://github.com/microsoft/brt/issues/4434
            # SO: https://stackoverflow.com/questions/52185507/pickle-and-decorated-classes-picklingerror-not-the-same-object

            # Store the inner class. The wrapped class couldn't be properly pickled.
            type_ = cloudpickle.dumps(type(self).__wrapped__)

            # in case they have customized ``__getstate__``.
            if hasattr(self, "__getstate__"):
                obj_ = self.__getstate__()
            else:
                obj_ = self.__dict__

            # Pickle can't handle type objects.
            if "_brt_symbol" in obj_:
                obj_["_brt_symbol"] = cloudpickle.dumps(obj_["_brt_symbol"])

            return _pickling_object, (type_, kw_only, obj_)

    _copy_class_wrapper_attributes(base, wrapper)

    return wrapper


def _trace_netlet_func(func, kw_only):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # similar to class, store parameters here
        args, kwargs = _formulate_arguments(func, args, kwargs, kw_only)

        # it's not clear whether this wrapper can handle all the types in python
        # There are many cases here: https://docs.python.org/3/reference/datamodel.html
        # but it looks that we have handled most commonly used cases
        res = func(
            *[_argument_processor(arg) for arg in args],
            **{kw: _argument_processor(arg) for kw, arg in kwargs.items()},
        )

        if res is None:
            # don't call super, makes no sense.
            # an empty serializable object is "none". Don't check it though.
            res = SerializableObject(func, args, kwargs, call_super=False)
        elif hasattr(res, "__class__") and hasattr(res, "__dict__"):
            # is a class, inject interface directly
            # need to be done before primitive types because there could be inheritance here.
            if not getattr(type(res), "_traced", False):
                _make_class_traceable(type(res), False)  # in-place
            res = inject_trace_info(res, func, args, kwargs)
        elif isinstance(res, (collections.abc.Callable, types.ModuleType, IOBase)):
            raise TypeError(
                f"Try to add trace info to {res}, but functions and modules are not supported."
            )
        elif isinstance(
            res,
            (
                numbers.Number,
                collections.abc.Sequence,
                collections.abc.Set,
                collections.abc.Mapping,
            ),
        ):
            # handle primitive types like int, str, set, dict, tuple
            # NOTE: simple types including none, bool, int, float, list, tuple, dict
            # will be directly captured by python json encoder
            # and thus not possible to restore the trace parameters after dump and reload.
            # this is a known limitation.
            new_type = _make_class_traceable(type(res), True)
            res = new_type(res)  # re-creating the object
            res = inject_trace_info(res, func, args, kwargs)
        else:
            raise TypeError(
                f'Try to add trace info to {res}, but the type "{type(res)}" is unknown. '
                "Please file an issue at https://github.com/Raphael-Hao/brainstorm/issues"
            )

        return res

    return wrapper
