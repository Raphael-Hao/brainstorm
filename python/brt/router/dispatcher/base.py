# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.


import abc

import numpy as np


class Dispatcher(abc.ABC):
    def __init__(self, route_num, gran_dim, transform, reduction):
        self.route_num = route_num
        self.gran_dim = gran_dim
        self.route_shape = list(gran_dim)
        self.route_size = gran_dim if isinstance(gran_dim, int) else np.prod(gran_dim)
        self.transform = transform
        self.reduction = reduction

    def dispatch(self, *inputs):
        raise NotImplementedError

    def combine(self, *inputs):
        raise NotImplementedError
