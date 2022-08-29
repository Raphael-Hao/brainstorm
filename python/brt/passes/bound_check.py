# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from brt.passes.base import PassBase, register_pass


@register_pass("dead_path_eliminate")
class DeadPathEliminatePass(PassBase):
    @classmethod
    def run_on_graph(cls):
        pass


