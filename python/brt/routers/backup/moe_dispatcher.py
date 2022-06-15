# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from tutel.impls.fast_dispatch import GatingEncoder, GatingDecoder, extract_critical
from .base import Dispatcher


class MoEDispatcher(Dispatcher):
    def __init__(self, route_num):
        super().__init__(route_num)

    def dispatch(self, inputs, route_indices):
        raise NotImplementedError

    def combine(self, *inputs):
        raise NotImplementedError


class TutelMoEDispatcher(MoEDispatcher):
    def __init__(self, route_num):
        super().__init__(route_num)


    def dispatch(self, *inputs):
        raise NotImplementedError

    def combine(self, *inputs):
        raise NotImplementedError
