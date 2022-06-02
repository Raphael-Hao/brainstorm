# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch.nn as nn
from brt.primitive import router


@router
class BaseRouter(nn.Module):
    def __init__(self, route_num: int):
        """_summary_

        Args:
            route_num (int): number of src or dst for routing
            gran_dim (_type_, optional): routing granularity. should be a int or a list of int.
        """
        super().__init__()
        self.route_num = route_num

    def route(self, *inputs):
        raise NotImplementedError

    def symbolic_route(self, *inputs):
        raise NotImplementedError
