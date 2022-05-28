# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import json
import re
from typing import Dict, List, Union


def remove_empty_lines(code: str) -> str:
    return re.sub(r"\n\s*\n", "\n", code)


def remove_comments(code: str) -> str:
    code = re.sub(re.compile("/\*.*?\*/", re.DOTALL), "", code)
    code = re.sub(re.compile(r"//.*?\n"), "", code)
    return code


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
    parameters: Dict[str, Union[Union[int, float], List[Union[int, float]]]],
) -> str:
    identifier_dict = {}
    identifier_dict["op_type"] = op_type
    identifier_dict["input_infos"] = input_infos
    identifier_dict["output_infos"] = output_infos
    identifier_dict["parameters"] = parameters
    identifier = json.dumps(identifier_dict)
    return identifier


def make_fused_identifier(identifiers):
    fused_indentifier_dict = {}
    for i, identifier in enumerate(identifiers):
        identifier_dict = json.loads(identifier)
        fused_indentifier_dict.update("fused_" + str(i), identifier_dict)
    fused_identifier = json.dumps(fused_indentifier_dict)
    return fused_identifier
