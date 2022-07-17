# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from brt.router.base import RouterBase


class SwinMoEScatterRouter(RouterBase):
    def __init__(self):
        super().__init__()
