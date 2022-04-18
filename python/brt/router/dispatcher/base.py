# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.


import abc


class Dispatcher(abc.ABC):
    def __init__(self, route_num):
        self.route_num = route_num

    def dispatch(self, *inputs):
        raise NotImplementedError

    def combine(self, *inputs):
        raise NotImplementedError
