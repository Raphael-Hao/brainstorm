# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import re
from typing import Dict, List, Union


def remove_empty_lines(code: str) -> str:
    return re.sub(r"\n\s*\n", "\n", code)


def check_if_pointer(param_type: str) -> bool:
    return re.search(r"\w+\s*\*", param_type) is not None


def make_identifier(
    op_type,
    input_infos: Dict[str, List[int]],
    output_infos: Dict[str, List[int]],
    parameters: Dict[str, List[Union[int, float]]],
) -> str:
    identifier = op_type
    identifier += "{"
    identifier += ",".join(
        "[" + ",".join(str(dim) for dim in shape) + "]"
        for shape in input_infos.values()
    )
    identifier += "};{"
    identifier += ",".join(
        "[" + ",".join(str(dim) for dim in shape) + "]"
        for shape in output_infos.values()
    )
    identifier += "};{"
    identifier += ",".join(
        "[" + ",".join(str(dim) for dim in parameter) + "]"
        if isinstance(parameter, list)
        else str(parameter)
        for parameter in parameters.values()
    )
    identifier += "}"
    return identifier


def make_attributes(param_type: str, param_name: str) -> str:
    return f"{make_identifier(param_type, param_name)} = {make_identifier(param_type, param_name)}"
