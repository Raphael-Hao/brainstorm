# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
from brt.runtime import Registry


class PassBase:
    def __init__(self) -> None:
        self.tracers = []

    def analyze(self, flow: torch.Tensor) -> None:
        pass

    def ru_on_graph(self, graph) -> None:
        pass

    def finalize(self) -> None:
        pass

def register_pass(pass_class: type) -> None:
    return Registry.register_sub_cls(pass_class, PassBase)
