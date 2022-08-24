# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
from brt.runtime import Registry


class PassBase:

    @classmethod
    def analyze(flow: torch.Tensor) -> None:
        pass

    @staticmethod
    def ru_on_graph(self, graph) -> None:
        pass

    @staticmethod
    def finalize(self) -> None:
        pass


def register_pass(pass_class: type) -> None:
    return Registry.register_sub_cls(pass_class, PassBase)
