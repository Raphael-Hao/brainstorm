# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from brt.runtime import Registry


class PassBase:
    def __init__(self) -> None:
        self.tracers = []

    def finalize(self) -> None:
        pass


def register_pass(pass_class: type) -> None:
    return Registry.register_cls(pass_class, PassBase)
