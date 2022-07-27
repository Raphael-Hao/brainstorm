# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from brt.transform.base import PassBase, register_pass


@register_pass("horiz_fuse")
class HorizFusePass(PassBase):
    pass
