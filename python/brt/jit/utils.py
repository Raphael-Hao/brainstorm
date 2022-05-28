# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import re
from typing import Dict, List, Union


def remove_empty_lines(code: str) -> str:
    return re.sub(r"\n\s*\n", "\n", code)


def check_if_pointer(param_type: str) -> bool:
    return re.search(r"\w+\s*\*", param_type) is not None


def make_func_name(
    op_type,
    input_infos: Dict[str, List[int]],
    output_infos: Dict[str, List[int]],
    parameters: Dict[str, Union[Union[int, float], List[Union[int, float]]]],
) -> str:
    func_name = op_type
    func_name += "__"
    func_name += "_".join(
        f"{name}_" + "_".join(str(dim) for dim in shape)
        for name, shape in input_infos.items()
    )
    func_name += "__"
    func_name += "_".join(
        f"{name}_" + "_".join(str(dim) for dim in shape)
        for name, shape in output_infos.items()
    )
    func_name += "__"
    func_name += "_".join(
        f"{name}_" + "_".join(str(dim) for dim in parameter)
        if isinstance(parameter, list)
        else f"{name}_" + str(parameter)
        for name, parameter in parameters.items()
    )
    return func_name


def make_identifier(
    op_type,
    input_infos: Dict[str, List[int]],
    output_infos: Dict[str, List[int]],
    parameters: Dict[str, List[Union[int, float]]],
) -> str:
    identifier = op_type
    identifier += "{"
    identifier += ",".join(
        f"{name}:[" + ",".join(str(dim) for dim in shape) + "]"
        for name, shape in input_infos.items()
    )
    identifier += "};{"
    identifier += ",".join(
        f"{name}:[" + ",".join(str(dim) for dim in shape) + "]"
        for name, shape in output_infos.items()
    )
    identifier += "};{"
    identifier += ",".join(
        f"{name}:[" + ",".join(str(dim) for dim in parameter) + "]"
        if isinstance(parameter, list)
        else f"{name}:[" + str(parameter)
        for name, parameter in parameters.items()
    )
    identifier += "}"
    return identifier
