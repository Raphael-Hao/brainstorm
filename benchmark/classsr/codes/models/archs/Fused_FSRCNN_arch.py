# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import functools
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
import torch

from models.archs.FSRCNN_arch import FSRCNN_net

from brt.jit import make_jit_kernel

class Fused_FSRCNN_net(torch.nn.Module):
    def __init__(self, raw: FSRCNN_net, bs: int):
        super(Fused_FSRCNN_net, self).__init__()
        self.bs = bs
        self.d = raw.d # 16
        self.s = raw.s # 12
        self.head_conv = make_jit_kernel(raw.head_conv, torch.randn(bs, 3, 32, 32).cuda())
        self.head_conv_args = [ # Conv2d+PReLU
            None,
            raw.head_conv[0].weight,
            None,
            raw.head_conv[0].bias,
            raw.head_conv[1].weight,
        ]
        self.body_conv = [
            make_jit_kernel(
                raw.body_conv[0],
                torch.randn(bs, self.d, 32, 32)
            ),
            make_jit_kernel(
                raw.body_conv[1],
                torch.randn(bs, self.s, 32, 32)
            ),
            make_jit_kernel(
                raw.body_conv[2],
                torch.randn(bs, self.s, 32, 32)
            ),
            make_jit_kernel(
                raw.body_conv[3],
                torch.randn(bs, self.s, 32, 32)
            ),
            make_jit_kernel(
                nn.Sequential(
                    raw.body_conv[4],
                    raw.body_conv[5],
                ),
                torch.randn(bs, self.s, 32, 32)
            ),
            make_jit_kernel(
                raw.body_conv[6],
                torch.randn(bs, self.d, 32, 32)
            ),
        ]
        self.body_conv_args = [
            [ # 0 Conv2d+PReLU 
                None,
                raw.body_conv[0].weight,
                None,
                raw.body_conv[0].bias,
                raw.body_conv[1].weight,
            ],
            [ # 1 Conv2d 
                None,
                raw.body_conv[1].weight,
                None,
                raw.body_conv[1].bias,
            ],
            [ # 2 Conv2d 
                None,
                raw.body_conv[2].weight,
                None,
                raw.body_conv[2].bias,
            ],
            [ # 3 Conv2d 
                None,
                raw.body_conv[3].weight,
                None,
                raw.body_conv[3].bias,
            ],
            [ # 4 Conv2d+PReLU 
                None,
                raw.body_conv[4].weight,
                None,
                raw.body_conv[4].bias,
                raw.body_conv[5].weight,
            ],
            [ # 5 Conv2d+PReLU 
                None,
                raw.body_conv[6].weight,
                None,
                raw.body_conv[6].bias,
                raw.body_conv[6].weight,
            ],
        ]
        self.tail_conv = make_jit_kernel(raw.tail_conv, torch.randn(bs, self.d, 32, 32))
        self.tail_conv_args =  [ # ConvTranspose2d
            raw.tail_conv.weight,
            None,
            raw.tail_conv.bias,
            None,
        ]
    def forward(self, x):
        self.head_conv_args[0] = x
        self.head_conv_args[2] = torch.empty(self.bs, self.d, 32 ,32).cuda()
        self.head_conv(*self.head_conv_args)
        x = self.head_conv_args[2]
        self.head_conv_args[0] = None
        self.head_conv_args[2] = None
        
        self.body_conv_args[0][0] = x
        self.body_conv_args[0][2] = torch.empty(self.bs, self.s, 32 ,32).cuda()
        self.body_conv[0](*self.head_conv_args[0])
        x = self.body_conv_args[0][2]
        self.body_conv_args[0][0] = None
        self.body_conv_args[0][2] = None
        
        self.body_conv_args[1][0] = x
        self.body_conv_args[1][2] = torch.empty(self.bs, self.s, 32 ,32).cuda()
        self.body_conv[1](*self.head_conv_args[1])
        x = self.body_conv_args[0][2]
        self.body_conv_args[1][0] = None
        self.body_conv_args[1][2] = None
        
        self.body_conv_args[2][0] = x
        self.body_conv_args[2][2] = torch.empty(self.bs, self.s, 32 ,32).cuda()
        self.body_conv[2](*self.head_conv_args[2])
        x = self.body_conv_args[0][2]
        self.body_conv_args[2][0] = None
        self.body_conv_args[2][2] = None
        
        self.body_conv_args[3][0] = x
        self.body_conv_args[3][2] = torch.empty(self.bs, self.s, 32 ,32).cuda()
        self.body_conv[3](*self.head_conv_args[3])
        x = self.body_conv_args[0][2]
        self.body_conv_args[3][0] = None
        self.body_conv_args[3][2] = None
        
        self.body_conv_args[4][0] = x
        self.body_conv_args[4][2] = torch.empty(self.bs, self.s, 32 ,32).cuda()
        self.body_conv[4](*self.head_conv_args[4])
        x = self.body_conv_args[0][2]
        self.body_conv_args[4][0] = None
        self.body_conv_args[4][2] = None
        
        self.body_conv_args[5][0] = x
        self.body_conv_args[5][2] = torch.empty(self.bs, self.s, 32 ,32).cuda()
        self.body_conv[5](*self.head_conv_args[5])
        x = self.body_conv_args[0][2]
        self.body_conv_args[5][0] = None
        self.body_conv_args[5][2] = None

        
        self.tail_conv_args[0] = x
        self.tail_conv_args[2] = torch.empty(self.bs, 3, 32 ,32).cuda()
        self.tail_conv(*self.tail_conv_args)
        x = self.tail_conv_args[2]
        self.tail_conv_args[0] = None

        return x
        