# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Callable, Dict, List, Type, Union, TypeVar
import inspect
import torch
from brt.runtime import Registry


def register_leaf_node(leaf_node_cls):
    if not issubclass(leaf_node_cls, torch.nn.Module):
        raise ValueError(
            f"{leaf_node_cls} is not a subclass of torch.nn.Module, it cannot be registered as a leaf node class."
        )
    global_register_func = Registry.register_cls("leaf_node")

    return global_register_func(leaf_node_cls)


def is_leaf_node(cls_or_instance) -> bool:

    if not inspect.isclass(cls_or_instance):
        leaf_node_cls = cls_or_instance.__class__
    else:
        leaf_node_cls = cls_or_instance

    return Registry.cls_exists_and_registered(leaf_node_cls, "leaf_node")
