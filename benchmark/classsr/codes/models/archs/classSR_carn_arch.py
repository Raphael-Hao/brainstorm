import functools
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
import torch
from models.archs.CARN_arch import CARN_M
import numpy as np
import time

from brt.router import ScatterRouter, GatherRouter


class ClassSR(nn.Module):
    def __init__(self, in_nc=3, out_nc=3):
        super(ClassSR, self).__init__()
        self.upscale = 4
        self.classifier = Classifier()
        self.net1 = CARN_M(
            in_nc=in_nc, out_nc=out_nc, nf=36, scale=4, multi_scale=False, group=4
        )
        self.net2 = CARN_M(
            in_nc=in_nc, out_nc=out_nc, nf=52, scale=4, multi_scale=False, group=4
        )
        self.net3 = CARN_M(
            in_nc=in_nc, out_nc=out_nc, nf=64, scale=4, multi_scale=False, group=4
        )
        self.scatter_router = ScatterRouter(
            protocol_type="topk", protocol_kwargs={"top_k": 1}
        )
        self.gather_router = GatherRouter(fabric_type="combine")

    def forward(self, x, is_train=False):

        weights = self.classifier(x)
        sr_x = self.scatter_router(x, weights)
        y = [self.net1(sr_x[0]), self.net2(sr_x[1]), self.net3(sr_x[2])]
        gr_x = self.gather_router(y)
        return gr_x, [yy.shape[0] for yy in y]

        # for i in range(len(x)):
        #     type = self.classifier(x[i].unsqueeze(0))

        #     flag = torch.max(type, 1)[1].data.squeeze()
        #     p = F.softmax(type, dim=1)
        #     # flag=np.random.randint(0,2)
        #     # flag=2
        #     if flag == 0:
        #         out = self.net1(x[i].unsqueeze(0))
        #     elif flag == 1:
        #         out = self.net2(x[i].unsqueeze(0))
        #     elif flag == 2:
        #         out = self.net3(x[i].unsqueeze(0))
        #     if i == 0:
        #         out_res = out
        #         type_res = p
        #     else:
        #         out_res = torch.cat((out_res, out), 0)
        #         type_res = torch.cat((type_res, p), 0)

        # return out_res, type_res


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.lastOut = nn.Linear(32, 3)

        # Condtion network
        self.CondNet = nn.Sequential(
            nn.Conv2d(3, 128, 4, 4),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 32, 1),
        )
        self.avgPool2d = nn.AvgPool2d(8)
        arch_util.initialize_weights([self.CondNet], 0.1)

    def forward(self, x):
        # assert x.shape[1:] == torch.Size([3, 32, 32]), x.shape
        out = self.CondNet(x)  # [bs, 32, 8, 8]
        out = self.avgPool2d(out)  # [bs, 32, 1, 1]
        out = out.view(-1, 32)  # [bs, 32]
        out = self.lastOut(out)  # [bs, 3]
        return out
