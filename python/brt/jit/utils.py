# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import re


def remove_empty_lines(code: str) -> str:
    return re.sub(r"\n\s*\n", "\n", code)
