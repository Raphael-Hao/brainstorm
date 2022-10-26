import math
import torch.nn as nn
import models.archs.arch_util as arch_util

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.channel = channel
        self.reduction = reduction
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self,
        conv,
        n_feat,
        kernel_size,
        reduction,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
        res_scale=1,
    ):

        super(RCAB, self).__init__()
        self.n_feat = n_feat
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks
    ):
        super(ResidualGroup, self).__init__()
        self.n_feat = n_feat
        modules_body = []
        modules_body = [
            RCAB(
                conv,
                n_feat,
                kernel_size,
                reduction,
                bias=True,
                bn=False,
                act=nn.ReLU(True),
                res_scale=1,
            )
            for _ in range(n_resblocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):
        self.n_feat = n_feat
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feat))
                if act:
                    m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if act:
                m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    """modified RCAN"""

    def __init__(
        self,
        n_resgroups,
        n_resblocks,
        n_feats,
        res_scale,
        n_colors,
        rgb_range,
        scale,
        reduction,
        conv=arch_util.default_conv,
    ):
        super(RCAN, self).__init__()

        self.n_resgroups = n_resgroups
        self.n_resblocks = n_resblocks
        self.n_feat = n_feats
        self.res_scale = res_scale
        self.n_colors = n_colors
        self.rgb_range = rgb_range
        self.kernel_size = 3
        self.scale = scale
        self.reduction = reduction
        act = nn.ReLU(True)

        # RGB mean for DIV2K
        self.rgb_mean = (0.4488, 0.4371, 0.4040)
        self.rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = arch_util.MeanShift(self.rgb_range, self.rgb_mean, self.rgb_std)

        # define head module
        modules_head = [conv(self.n_colors, self.n_feat, self.kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv,
                self.n_feat,
                self.kernel_size,
                self.reduction,
                act=act,
                res_scale=self.res_scale,
                n_resblocks=self.n_resblocks,
            )
            for _ in range(self.n_resgroups)
        ]

        modules_body.append(conv(self.n_feat, self.n_feat, self.kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(conv, self.scale, self.n_feat, act=False),
            conv(self.n_feat, self.n_colors, self.kernel_size),
        ]

        self.add_mean = arch_util.MeanShift(
            self.rgb_range, self.rgb_mean, self.rgb_std, 1
        )

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

        arch_util.initialize_weights([self.head, self.body, self.tail], 0.1)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x
