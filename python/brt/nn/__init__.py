# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.


import copy
import inspect
import warnings
from pathlib import Path

import torch
import torch.nn as nn

# To make auto-completion happy, we generate a _nn.py that lists out all the classes.

nn_cache_file_path = Path(__file__).parent / "_nn.py"
_nn_module_files = []

CACHE_VALID = False

if nn_cache_file_path.exists():
    from . import _nn  # pylint: disable=no-name-in-module

    # valid only when torch version match
    if _nn._torch_version == torch.__version__:
        CACHE_VALID = True
    else:
        for module_name in _nn.all_module_names:
            # delete module_fpath if exist
            module_fpath = Path(__file__).parent / (module_name + ".py")
            if module_fpath.exists():
                module_fpath.unlink()


if not CACHE_VALID:
    _NO_WRAP_CLASSES = [
        # not an nn.Module
        "Parameter",
        "ParameterList",
        "UninitializedBuffer",
        "UninitializedParameter",
        # arguments are special
        "Module",
        "Sequential",
        # utilities
        "Container",
        "DataParallel",
    ]

    _WRAP_WITHOUT_TAG_CLASSES = [
        # special support on graph engine
        "ModuleList",
        "ModuleDict",
    ]

    code = [
        "# This file is auto-generated to make auto-completion work.",
        "# When pytorch version does not match, it will get automatically updated.",
        "# pylint: skip-file",
        f'_torch_version = "{torch.__version__}"',
        "import torch.nn as nn",
        "from brt.prim import netlet",
    ]

    obj_common_header = [
        "# This file is auto-generated to make auto-completion work.",
        "# When pytorch version does not match, it will get automatically updated.",
        "# pylint: skip-file",
    ]

    all_cls_fn_names = []
    all_module_names = []

    # Add modules, classes, functions in torch.nn into this module.
    for name, obj in inspect.getmembers(torch.nn):
        if inspect.isclass(obj):
            if name in _NO_WRAP_CLASSES:
                code.append(f"{name} = nn.{name}")
            elif not issubclass(obj, nn.Module):
                # It should never go here
                # We did it to play safe
                warnings.warn(
                    f"{obj} is found to be not a nn.Module, which is unexpected. "
                    "It means your PyTorch version might not be supported.",
                    RuntimeWarning,
                )
                code.append(f"{name} = nn.{name}")
            elif name in _WRAP_WITHOUT_TAG_CLASSES:
                code.append(f"{name} = netlet(nn.{name}, netlet_tag=False)")
            else:
                code.append(f"{name} = netlet(nn.{name})")

            all_cls_fn_names.append(name)

        elif inspect.isfunction(obj):
            code.append(f"{name} = nn.{name}")  # no modification
            all_cls_fn_names.append(name)
        elif inspect.ismodule(obj):
            # no modification
            obj_file_name = Path(__file__).parent / f"{name}.py"
            obj_code = copy.deepcopy(obj_common_header)
            obj_code.append(
                f"from torch.nn.{name} import * # pylint: disable=wildcard-import, unused-wildcard-import"
            )
            all_module_names.append(name)
            with obj_file_name.open("w") as f:
                f.write("\n".join(obj_code))

    code.append(f"__all__ = {all_cls_fn_names}")
    code.append(f"all_module_names = {all_module_names}")

    with nn_cache_file_path.open("w") as fp:
        fp.write("\n".join(code))


# Import all modules from generated _nn.py

from . import _nn  # pylint: disable=no-name-in-module

__all__ = _nn.__all__
from ._nn import *  # pylint: disable=import-error, wildcard-import
