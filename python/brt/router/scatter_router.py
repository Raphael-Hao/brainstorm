#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from ctypes import Union
from typing import Tuple, List
import torch
import torch.nn.functional as F
import numpy as np
from .base import Router

__all__ = ["ScatterRouter", "RandomScatterRouter", "TopKScatterRouter"]


class ScatterRouter(Router):
    def __init__(self, route_num: int):
        super().__init__()
        self.route_num = route_num

    def route(
        self, *inputs
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Size]:
        raise NotImplementedError

    def inspect_inputs(self, inputs: Union[torch.Tensor, List[torch.Tensor]]):
        if isinstance(inputs, torch.Tensor):
            inputs_size = inputs.shape[0]
            inputs_shape = inputs.shape
        elif isinstance(inputs, list):
            inputs_size = len(inputs)
            inputs_shape = inputs_size
        else:
            raise ValueError("inputs must be a list of tensor or a tensor")
        return inputs_size, inputs_shape

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
        inputs_size, inputs_shape = self.inspect_inputs(inputs)
        route_targets = np.random.randint(0, self.route_num, (inputs_size,))
        route_results = [None] * self.route_num
        reverse_indices = [None] * self.route_num
        for i in range(self.route_num):
            route_indices = np.nonzero(route_targets == i)[0]
            if len(route_indices) > 0:
                tmp_results = [inputs[j] for j in route_indices]
                route_results[i] = torch.stack(tmp_results)
                reverse_indices[i] = route_indices
        return route_results, reverse_indices, inputs_shape


class TopKScatterRouter(ScatterRouter):
    def __init__(self, route_num: int, grain_dim: int, k=2, top_fn=None):
        super().__init__(route_num=route_num)
        self.grain_dim = grain_dim
        assert self.grain_dim > 0, f"grain dim {self.grain_dim} is not valid"
        self.k = min(k, route_num)
        assert self.k > 0, f"Top-K value {self.k} is not valid"
        if top_fn == None:
            self.top_fn = torch.nn.Linear(self.grain_dim, self.route_num, bias=False)
        else:
            self.top_fn = top_fn

    def route(
        self, *inputs
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Size]:
        inputs_size, inputs_shape = self.inspect_inputs(inputs)

        route_results = [None] * self.route_num
        reverse_indices = [None] * self.route_num
        for i in range(self.route_num):
            route_indices = np.nonzero(route_targets == i)[0]
            if len(route_indices) > 0:
                tmp_results = [inputs[j] for j in route_indices]
                route_results[i] = torch.stack(tmp_results)
                reverse_indices[i] = route_indices
        return route_results, reverse_indices, inputs_shape

    def topk_select(self, inputs):
        logits = self.top_fn(inputs)
        gates = F.softmax(logits, dim=1)
        