#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch

from .base import Router


class GatherRouter(Router):
    def __init__(self, select_fn=None, support_batch=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.select_fn = select_fn
        self.support_batch = support_batch

    def route(self, *inputs):
        return inputs[0]

    def record(self):
        self.active_counter += 1

    def register_router(self):
        pass


class RandomGatherRouter(GatherRouter):
    def __init__(self, support_batch=False, *args, **kwargs):
        super().__init__(self.random_select, support_batch, *args, **kwargs)

    def random_select(self, input):
        return torch.randint(0, input.size(0), (input.size(0),))


class TopKGatherRouter(GatherRouter):
    def __init__(self, support_batch=False, *args, **kwargs):
        super().__init__(self.topk_select, support_batch, *args, **kwargs)

    def topk_select(self, input):
        return torch.topk(input, self.top, dim=0)[1]
