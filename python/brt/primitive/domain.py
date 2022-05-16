# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import inspect
from typing import TypeVar, Union

# import brt.logger as logger
import cloudpickle
import torch

from .base import (
    SerializableObject,
    Traceable,
    _copy_class_wrapper_attributes,
    _formulate_arguments,
    _is_function,
    _pickling_object,
    _unwrap_metaclass,
    torchscript_patch,
)
from .helper import (
    check_wrapped,
    is_domain,
    is_netlet,
    is_router,
    is_wrapped_with_trace,
)

T = TypeVar("T")


def unwrap_netlet(m):
    if is_netlet(m):
        m.using_brt_forward = False
        m.forward = m.pt_forward
    return m


def unwrap_redundant_netlet(m):
    if is_domain(m):
        # logger.Debug("unwrap redundant netlet because it is a domain")
        unwrap_netlet(m)
    # check if router in children
    if_redundant = True
    for child in m.children():
        if is_router(child):
            if_redundant = False
            break
    if if_redundant:
        # logger.Debug("unwrap redundant netlet due to no router")
        for child in m.children():
            unwrap_netlet(child)
    for child in m.children():
        unwrap_redundant_netlet(child)
    return m


def optimize_switch(m, switch=True):
    for child in m.children():
        optimize_switch(child, switch=switch)
    if is_router(m):
        m._brt_optimize = switch
        if m._brt_optimize:
            m.forward = m.symbolic_route
        else:
            m.forward = m.route
    return m


def domain(cls: T, domain_tag=True) -> T:
    """Decorator for annotating the whole graph

    Args:
        cls (T): _description_

    Returns:
        T: _description_
    """
    if check_wrapped(cls, "domain"):
        return cls

    cls = trace_domain(cls)

    cls._brt_domain = domain_tag
    torchscript_patch(cls)
    return cls


def trace_domain(
    cls_or_func: T = None, *, kw_only: bool = True, inheritable: bool = False
) -> Union[T, Traceable]:
    def wrap(cls_or_func):
        # already annotated, do nothing
        if is_wrapped_with_trace(cls_or_func):
            return cls_or_func
        if isinstance(cls_or_func, type):
            cls_or_func = _trace_domain_cls(
                cls_or_func, kw_only, inheritable=inheritable
            )
        elif _is_function(cls_or_func):
            # cls_or_func = _trace_netlet_func(cls_or_func, kw_only)
            raise NotImplementedError(
                "Tracing function is not implemented yet for domain."
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


def _trace_domain_cls(
    base: T = None, kw_only: bool = True, call_super=True, inheritable=False
):
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
            unwrap_redundant_netlet(self)

        def optimize(self, switch=True):
            return optimize_switch(self, switch=switch)

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
