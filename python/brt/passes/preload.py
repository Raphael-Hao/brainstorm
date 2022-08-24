# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from brt.passes.base import PassBase, register_pass

@register_pass("weight_preload")
class WeightPreloadPass(PassBase):
    pass