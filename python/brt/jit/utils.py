# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import re


def remove_empty_lines(code: str) -> str:
    return re.sub(r"\n\s*\n", "\n", code)

def check_if_pointer(param_type: str) -> bool:
    return re.search(r"\w+\s*\*", param_type) is not None
