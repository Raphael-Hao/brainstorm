# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

import torch


def inject_source(source: str) -> int: ...
def invoke(inputs: List[torch.Tensor], extra: List[int], ctx: int) -> None: ...

