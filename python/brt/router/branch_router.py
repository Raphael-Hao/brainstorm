# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from .base import Router


class BranchRouter(Router):
    def __init__(self, select_fn=None, support_batch=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.select_fn = select_fn
        self.support_batch = support_batch

    def route(self, *inputs):
        if self.support_batch:
            indice_matrix = self.select_fn(inputs[0])
        else:
            indices = []
            for input in inputs:
                indices.append(self.select_fn(input))

    def record(self):
        self.active_counter += 1

    def register_router(self):
        pass
