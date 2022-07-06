# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from brt.routers.router import RouterBase


class SwinMoEScatterRouter(RouterBase):
    def __init__(self, path_num: int):
        super().__init__(path_num)
