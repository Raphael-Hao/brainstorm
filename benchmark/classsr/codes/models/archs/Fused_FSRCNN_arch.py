# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.archs.FSRCNN_arch import FSRCNN_net

from brt.jit import make_jit_kernel


class Fused_FSRCNN_net(torch.nn.Module):
    def __init__(self, raw: FSRCNN_net, bs: int):
        super(Fused_FSRCNN_net, self).__init__()
        self.bs = bs
        self.d = raw.d  # 16
        self.s = raw.s  # 12
        for name, tensor in raw.named_parameters():
            if "tail" not in name:
                self.register_parameter(name.replace(".", "_"), tensor)
        for name, tensor in raw.named_buffers():
            if "tail" not in name:
                self.register_buffer(name.replace(".", "_"), tensor)
        self.head_conv = make_jit_kernel(
            raw.head_conv, torch.randn(bs, 3, 32, 32, device="cuda")
        )
        self.head_conv_args = [  # Conv2d+PReLU
            None,
            self.head_conv_0_weight,
            None,
            self.head_conv_0_bias,
            self.head_conv_1_weight,
        ]
        self.body_conv = [
            make_jit_kernel(
                raw.body_conv[0], torch.randn(bs, self.d, 32, 32, device="cuda")
            ),
            make_jit_kernel(
                raw.body_conv[1], torch.randn(bs, self.s, 32, 32, device="cuda")
            ),
            make_jit_kernel(
                raw.body_conv[2], torch.randn(bs, self.s, 32, 32, device="cuda")
            ),
            make_jit_kernel(
                raw.body_conv[3], torch.randn(bs, self.s, 32, 32, device="cuda")
            ),
            make_jit_kernel(
                nn.Sequential(
                    raw.body_conv[4],
                    raw.body_conv[5],
                ),
                torch.randn(bs, self.s, 32, 32, device="cuda"),
            ),
            make_jit_kernel(
                raw.body_conv[6], torch.randn(bs, self.s, 32, 32, device="cuda")
            ),
        ]
        self.body_conv_args = [
            [  # 0 Conv2d+PReLU
                None,
                self.body_conv_0_0_weight,
                None,
                self.body_conv_0_0_bias,
                self.body_conv_0_1_weight,
            ],
            [  # 1 Conv2d
                None,
                self.body_conv_1_weight,
                None,
                self.body_conv_1_bias,
            ],
            [  # 2 Conv2d
                None,
                self.body_conv_2_weight,
                None,
                self.body_conv_2_bias,
            ],
            [  # 3 Conv2d
                None,
                self.body_conv_3_weight,
                None,
                self.body_conv_3_bias,
            ],
            [  # 4 Conv2d+PReLU
                None,
                self.body_conv_4_weight,
                None,
                self.body_conv_4_bias,
                self.body_conv_5_weight,
            ],
            [  # 5 Conv2d+PReLU
                None,
                self.body_conv_6_0_weight,
                None,
                self.body_conv_6_0_bias,
                self.body_conv_6_1_weight,
            ],
        ]
        self.tail_conv = raw.tail_conv

    def forward(self, x):

        self.head_conv_args[0] = x
        self.head_conv_args[2] = torch.empty(self.bs, self.d, 32, 32, device="cuda")
        self.head_conv(*self.head_conv_args)
        x = self.head_conv_args[2]
        self.head_conv_args[0] = None
        self.head_conv_args[2] = None

        self.body_conv_args[0][0] = x
        self.body_conv_args[0][2] = torch.empty(self.bs, self.s, 32, 32, device="cuda")
        self.body_conv[0](*self.body_conv_args[0])
        x = self.body_conv_args[0][2]
        self.body_conv_args[0][0] = None
        self.body_conv_args[0][2] = None

        self.body_conv_args[1][0] = x
        self.body_conv_args[1][2] = torch.empty(self.bs, self.s, 32, 32, device="cuda")
        self.body_conv[1](*self.body_conv_args[1])
        x = self.body_conv_args[1][2]
        self.body_conv_args[1][0] = None
        self.body_conv_args[1][2] = None

        self.body_conv_args[2][0] = x
        self.body_conv_args[2][2] = torch.empty(self.bs, self.s, 32, 32, device="cuda")
        self.body_conv[2](*self.body_conv_args[2])
        x = self.body_conv_args[2][2]
        self.body_conv_args[2][0] = None
        self.body_conv_args[2][2] = None

        self.body_conv_args[3][0] = x
        self.body_conv_args[3][2] = torch.empty(self.bs, self.s, 32, 32, device="cuda")
        self.body_conv[3](*self.body_conv_args[3])
        x = self.body_conv_args[3][2]
        self.body_conv_args[3][0] = None
        self.body_conv_args[3][2] = None

        self.body_conv_args[4][0] = x
        self.body_conv_args[4][2] = torch.empty(self.bs, self.s, 32, 32, device="cuda")
        self.body_conv[4](*self.body_conv_args[4])
        x = self.body_conv_args[4][2]
        self.body_conv_args[4][0] = None
        self.body_conv_args[4][2] = None

        self.body_conv_args[5][0] = x
        self.body_conv_args[5][2] = torch.empty(self.bs, self.d, 32, 32, device="cuda")
        self.body_conv[5](*self.body_conv_args[5])
        x = self.body_conv_args[5][2]
        self.body_conv_args[5][0] = None
        self.body_conv_args[5][2] = None

        x = self.tail_conv(x)

        return x
