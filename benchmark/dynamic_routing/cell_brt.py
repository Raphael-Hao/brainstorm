# encoding: utf-8
# network file -> build Cell for Dynamic Backbone
# @author: yanwei.li

import sys
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ops import OPS, Conv2dNormAct, Identity, kaiming_init_module

print(sys.path)

from brt.router import ScatterRouter, GatherRouter


__all__ = ["Mixed_OP", "Cell"]

# soft gate for path choice
def soft_gate(x, x_t=None, momentum=0.1, is_update=False):
    if is_update:
        # using momentum for weight update
        y = (1 - momentum) * x.data + momentum * x_t
        tanh_value = torch.tanh(y)
        return F.relu(tanh_value), y.data
    else:
        tanh_value = torch.tanh(x)
        return F.relu(tanh_value)


# Scheduled Drop Path
def drop_path(x, drop_prob, layer_rate, step_rate):
    """
    :param x: input feature
    :param drop_prob: drop path prob
    :param layer_rate: current_layer/total_layer
    :param step_rate: current_step/total_step
    :return: output feature
    """
    if drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        keep_prob = 1.0 - layer_rate * (1.0 - keep_prob)
        keep_prob = 1.0 - step_rate * (1.0 - keep_prob)
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob)
        x.mul_(mask)
    return x


class Mixed_OP(nn.Module):
    """
    Sum up operations according to their weights.
    """

    def __init__(
        self,
        inplanes,
        outplanes,
        stride,
        cell_type,
        norm="",
        affine=True,
        input_size=None,
    ):
        super(Mixed_OP, self).__init__()
        self._ops = nn.ModuleList()
        # self.op_flops = []
        for key in cell_type:
            op = OPS[key](
                inplanes,
                outplanes,
                stride,
                norm_layer=norm,
                affine=affine,
                input_size=input_size,
            )
            self._ops.append(op)
            # self.op_flops.append(op.flops)
        # if IS_CALCU_FLOPS in locals() and IS_CALCU_FLOPS:
        #     self.real_flops = sum(op_flop for op_flop in self.op_flops)

    def forward(self, x):

        return sum(op(x) for op in self._ops)

    @property
    def flops(self):
        return self.real_flops.squeeze()


class Cell(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        norm,
        allow_up,
        allow_down,
        input_size,
        cell_type,
        cal_flops=True,
        using_gate=False,
        small_gate=False,
        gate_bias=1.5,
        affine=True,
    ):
        super(Cell, self).__init__()
        self.channel_in = C_in
        self.channel_out = C_out
        self.allow_up = allow_up
        self.allow_down = allow_down
        self.cal_flops = cal_flops
        self.using_gate = using_gate
        self.small_gate = small_gate

        self.cell_ops = Mixed_OP(
            inplanes=self.channel_in,
            outplanes=self.channel_out,
            stride=1,
            cell_type=cell_type,
            norm=norm,
            affine=affine,
            input_size=input_size,
        )

        if self.allow_up and self.allow_down:
            self.gate_num = 3
        elif self.allow_up or self.allow_down:
            self.gate_num = 2
        else:
            self.gate_num = 1

        self.cell_gather = GatherRouter(
            fabric_type="single_ptu_combine", fabric_kwargs={"sparse": True}
        )

        self.residual_scatter = ScatterRouter(
            protocol_type="batched_threshold",
            fabric_type="single_ptu_dispatch",
            protocol_kwargs={"threshold": 0.0001, "single_tpu": True},
            fabric_kwargs={
                "flow_num": 2,
                "route_logic": ["1d", "1d"],
                "transform": [False, False],
            },
        )

        self.threeway_scatter = ScatterRouter(
            dispatch_score=True,
            protocol_type="threshold",
            fabric_type="single_ptu_dispatch",
            protocol_kwargs={
                "threshold": 0.0001,
                "residual_path": -1,
                "single_tpu": True,
            },
        )

        # resolution keep
        self.res_keep = nn.ReLU()
        # resolution up and dim down
        if self.allow_up:
            self.res_up = nn.Sequential(
                nn.ReLU(),
                Conv2dNormAct(
                    self.channel_out,
                    self.channel_out // 2,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                    # TODO: norm type
                    norm=nn.SyncBatchNorm(self.channel_out // 2),
                    activation=nn.ReLU(),
                ),
            )
            # using Kaiming init
            kaiming_init_module(self.res_up, mode="fan_in")
        # resolution down and dim up
        if self.allow_down:
            self.res_down = nn.Sequential(
                nn.ReLU(),
                Conv2dNormAct(
                    self.channel_out,
                    2 * self.channel_out,
                    kernel_size=1,
                    stride=2,
                    padding=0,
                    bias=False,
                    # TODO: norm type
                    norm=nn.SyncBatchNorm(self.channel_out * 2),
                    activation=nn.ReLU(),
                ),
            )
            # using Kaiming init
            kaiming_init_module(self.res_down, mode="fan_in")
        if self.using_gate:
            self.gate_conv_beta = nn.Sequential(
                Conv2dNormAct(
                    self.channel_in,
                    self.channel_in // 2,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                    # TODO: norm type
                    norm=nn.SyncBatchNorm(self.channel_in // 2),
                    activation=nn.ReLU(),
                ),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(
                    self.channel_in // 2,
                    self.gate_num,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                ),
            )
            if self.small_gate:
                input_size = input_size // 4
            # using Kaiming init and predefined bias for gate
            kaiming_init_module(self.gate_conv_beta, mode="fan_in", bias=gate_bias)
        else:
            self.register_buffer(
                "gate_weights_beta", torch.ones(1, self.gate_num, 1, 1).cuda()
            )

    def forward(self, h_l1: List[torch.Tensor]):
        """
        :param h_l1: # the former hidden layer output
        :return: current hidden cell result h_l
        """
        h_l1 = self.cell_gather(h_l1)
        if self.small_gate:
            h_l1_gate = F.interpolate(
                input=h_l1,
                scale_factor=0.25,
                mode="bilinear",
                align_corners=False,
            )
        else:
            h_l1_gate = h_l1
        gate_feat_beta = self.gate_conv_beta(h_l1_gate)
        gate_weights_beta = soft_gate(gate_feat_beta)
        gate_weights_beta = gate_weights_beta.view(
            gate_weights_beta.shape[0], self.gate_num
        )

        residual_h_l, residual_w_beta = self.residual_scatter(
            [h_l1, gate_weights_beta], gate_weights_beta
        )

        h_l = self.cell_ops(residual_h_l[1])

        route_h_l, route_weight = self.threeway_scatter(
            h_l, residual_w_beta[1].view(h_l.size(0), self.gate_num)
        )

        ## keep
        route_h_l_keep = self.res_keep(route_h_l[0])

        route_result_keep = (route_weight[0].view(-1, 1, 1, 1)) * route_h_l_keep

        ## up
        if self.allow_up:
            route_h_l_up = self.res_up(route_h_l[1])
            route_h_l_up = F.interpolate(
                input=route_h_l_up,
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            )
            route_result_up = route_h_l_up * (route_weight[1].view(-1, 1, 1, 1))

        ## down
        if self.allow_down:
            route_h_l_down = self.res_down(route_h_l[-1])
            route_result_down = route_h_l_down * (route_weight[-1].view(-1, 1, 1, 1))

        if self.allow_up and self.allow_down:
            return [
                [route_result_up],
                [route_result_keep, residual_h_l[0]],
                [route_result_down],
            ]
        elif self.allow_up:
            return [[route_result_up], [route_result_keep, residual_h_l[0]], []]
        elif self.allow_down:
            return [[], [route_result_keep, residual_h_l[0]], [route_result_down]]
        else:
            return [[], [route_result_keep, residual_h_l[0]], []]
