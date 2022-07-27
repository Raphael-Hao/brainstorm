# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.


import brt
from brt.transform.base import PassBase, register_pass

@register_pass("pipline")
class PipelinePass(PassBase):
    pass

@register_pass("sharded")
class ShardedPass(PassBase):
    pass
