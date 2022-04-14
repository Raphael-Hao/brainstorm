#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Tuple, List
import torch
import numpy as np
from .base import Router

__all__ = ["ScatterRouter", "RandomScatterRouter", "TopKScatterRouter"]


class ScatterRouter(Router):
    def __init__(self, route_num: int):
        super().__init__()
        self._route_num = route_num

    def route(
        self, *inputs
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Size]:
        raise NotImplementedError

    def record(self):
        self.active_counter += 1

    def register_router(self):
        pass


class RandomScatterRouter(ScatterRouter):
    def __init__(self, route_num: int):
        super().__init__(route_num=route_num)

    def route(
        self, inputs
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Size]:
        if isinstance(inputs, torch.Tensor):
            inputs_size = inputs.size(0)
            origin_shape = inputs.size()
        elif isinstance(inputs, list):
            inputs_size = len(inputs)
            origin_shape = inputs_size
        else:
            raise ValueError("inputs must be a list of tensor or a tensor")
        route_targets = np.random.randint(0, self._route_num, (inputs_size,))
        route_results = [None] * self._route_num
        reverse_indices = [None] * self._route_num
        for i in range(self._route_num):
            route_indices = np.nonzero(route_targets == i)[0]
            if len(route_indices) > 0:
                tmp_results = [inputs[j] for j in route_indices]
                route_results[i] = torch.stack(tmp_results)
                reverse_indices[i] = route_indices
        print(route_results)
        return route_results, reverse_indices, origin_shape


class TopKScatterRouter(ScatterRouter):
    def __init__(self, route_num: int, k: int):
        super().__init__(route_num=route_num)
        self._k = k

    def topk_select(self, inputs):
        return torch.topk(inputs, self.top, dim=0)[1]
