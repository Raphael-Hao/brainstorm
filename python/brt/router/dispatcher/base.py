# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.


import abc

import numpy as np


class Dispatcher(abc.ABC):
    def __init__(self, route_num, transform, reduction):
        self.route_num = route_num
        self.transform = transform
        self.reduction = reduction

    def dispatch(self, *inputs):
        raise NotImplementedError

    def combine(self, *inputs):
        raise NotImplementedError
