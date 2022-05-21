# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import re


def remove_empty_lines(code: str) -> str:
    return re.sub(r"\n\s*\n", "\n", code)


def check_if_pointer(param_type: str) -> bool:
    return re.search(r"\w+\s*\*", param_type) is not None


def make_key(param_type: str, param_name: str) -> str:
    return f"{param_type}_{param_name}"


def make_identifier(param_type: str, param_name: str) -> str:
    return f"{param_name}_{make_key(param_type, param_name)}"


def make_op_type(param_type: str, param_name: str) -> str:
    return f"{param_type} {make_identifier(param_type, param_name)}"


def make_attributes(param_type: str, param_name: str) -> str:
    return f"{make_identifier(param_type, param_name)} = {make_identifier(param_type, param_name)}"
