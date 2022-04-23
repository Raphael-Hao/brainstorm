# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from brt.primitive import router

from .base import BaseRouter


@router
class BranchRouter(BaseRouter):
    def __init__(self, select_fn=None, support_batch=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.select_fn = select_fn
        self.support_batch = support_batch

    def forward(self, *inputs):
        pass

    def record(self):
        self.active_counter += 1

    def register_router(self):
        pass
