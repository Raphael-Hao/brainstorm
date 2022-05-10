# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple

import torch


def inject_source(source: str) -> Tuple[str, int]: ...
def static_invoke(inputs: List[torch.Tensor], extra: List[int], ctx: int) -> None: ...
def hetero_invoke(inputs: List[torch.Tensor], extra: List[int], ctx: int) -> None: ...
def homo_invoke(
    inputs: List[torch.Tensor], weights: List[torch.Tensor], extra: List[int], ctx: int
) -> None: ...
def elastic_homo_invoke(
    inputs: List[torch.Tensor], weights: List[torch.Tensor], extra: List[int], ctx: int
) -> None: ...
