#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch

from .base import Router

__all__ = ['GatherRouter', 'RandomGatherRouter', 'TopKGatherRouter']

class GatherRouter(Router):
    def __init__(
        self,
        route_num: int,
    ):
        super().__init__()
        self._route_num = route_num

    def route(self, reverse_indices, *inputs):
        raise NotImplementedError

    def record(self):
        self.active_counter += 1

    def register_router(self):
        pass


class RandomGatherRouter(GatherRouter):
    def __init__(self, route_num: int):
        super().__init__(route_num=route_num)

    def route(self, reverse_indices, origin_shape, *inputs):
        assert (
            len(inputs) == self._route_num and len(reverse_indices) == self._route_num
        )
        if isinstance(origin_shape, int):
            route_size = origin_shape
        elif isinstance(origin_shape, torch.Size):
            route_size = origin_shape[0]
        else:
            raise ValueError("origin_shape must be a int or torch.Size")
        route_results = [[] for _ in range(route_size)]
        for i in range(self._route_num):
            if reverse_indices[i] is not None:
                for j in range(len(reverse_indices[i])):
                    route_results[reverse_indices[i][j]] = inputs[i][j]
        if isinstance(origin_shape, int):
            return route_results
        else:
            route_results = torch.stack(route_results).view(origin_shape)
            return route_results


class TopKGatherRouter(GatherRouter):
    def __init__(self, route_num: int, k: int):
        super().__init__(route_num=route_num)
        self._k = k
