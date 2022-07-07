# encoding: utf-8
# network file -> build Cell for Dynamic Backbone
# @author: yanwei.li

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from ops import OPS, Conv2dNormAct, Identity, kaiming_init_module

print(sys.path)

from brt.routers import ScatterRouter, GatherRouter, ProtoTensor

from macro import *

from typing import *

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
        self, inplanes, outplanes, stride, cell_type, norm="", affine=True, input_size=None,
    ):
        super(Mixed_OP, self).__init__()
        self._ops = nn.ModuleList()
        # self.op_flops = []
        for key in cell_type:
            op = OPS[key](
                inplanes, outplanes, stride, norm_layer=norm, affine=affine, input_size=input_size,
            )
            self._ops.append(op)
            # self.op_flops.append(op.flops)
        # if IS_CALCU_FLOPS in locals() and IS_CALCU_FLOPS:
        #     self.real_flops = sum(op_flop for op_flop in self.op_flops)

    def forward(self, x, is_drop_path=False, drop_prob=0.0, layer_rate=0.0, step_rate=0.0):
        if is_drop_path:
            y = []
            for op in self._ops:
                if not isinstance(op, Identity):
                    y.append(drop_path(op(x), drop_prob, layer_rate, step_rate))
                else:
                    y.append(op(x))
            return sum(y)
        else:
            # using sum up rather than random choose one branch.
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

        self.scatter_router = ScatterRouter(
            dst_num=self.gate_num,
            route_method="threshold",
            threshold=0,
            residual_dst=0,
            routing_gates=True,
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
            self.register_buffer("gate_weights_beta", torch.ones(1, self.gate_num, 1, 1).cuda())

    def forward(
        self,
        h_l1: Union[torch.Tensor, ProtoTensor],
        is_drop_path=False,
        drop_prob=0.0,
        layer_rate=0.0,
        step_rate=0.0,
    ):
        """
        :param h_l1: # the former hidden layer output
        :return: current hidden cell result h_l
        """
        drop_cell = False
        # drop the cell if input type is float
        # NOTE: change the branch condition
        # if not isinstance(h_l1, float):
        if h_l1.numel() != 0:
            # calculate soft conditional gate
            # gate_weights_beta = [keep(, up(, down)?)?]
            if self.using_gate:
                if self.small_gate:
                    h_l1_gate = F.interpolate(
                        input=h_l1, scale_factor=0.25, mode="bilinear", align_corners=False,
                    )
                else:
                    h_l1_gate = h_l1
                gate_feat_beta = self.gate_conv_beta(h_l1_gate)
                gate_weights_beta = soft_gate(gate_feat_beta)
            else:
                gate_weights_beta = self.gate_weights_beta
        else:
            drop_cell = True
        # use for inference
        if not self.training:
            if not drop_cell:
                drop_cell = gate_weights_beta.sum() < 0.0001
            if drop_cell:
                # result_list = [[0.0], [h_l1], [0.0]]
                # weights_list = [[0.0], [0.0], [0.0]]
                # ProtoTensor(
                #     data: tensor([], device='cuda:0')
                #     tag_stack: [tensor([], device='cuda:0', size=(0, 1), dtype=torch.int64)]
                #     load stack: [11])
                result_list = [
                    [torch.empty(0).to(h_l1.device)],
                    [h_l1],
                    [torch.empty(0).to(h_l1.device)],
                ]
                weights_list = [
                    [torch.empty(0).to(h_l1.device)],
                    [torch.empty(0).to(h_l1.device)],
                    [torch.empty(0).to(h_l1.device)],
                ]
                return (
                    result_list,
                    weights_list,
                )
            h_l = self.cell_ops(h_l1, is_drop_path, drop_prob, layer_rate, step_rate)
            # NOTE: brt, using for inference
            # route = [keep(, up(, down)?)?]
            route_h_l, route_weight = self.scatter_router(
                h_l, gate_weights_beta.view(h_l.size(0), self.gate_num)
            )
            route_weight = [x.view(-1, 1, 1, 1) for x in route_weight]
            result_list = [[], [], []]
            weights_list = [[], [], []]
            ## keep
            route_h_l_keep = self.res_keep(route_h_l[0])
            if isinstance(route_h_l[0], ProtoTensor):
                route_h_l1 = h_l1.index_select(0, route_weight[0].tag.squeeze())
            else:  # isinstance(h_l_route[0], torch.Tensor)
                route_h_l1 = h_l1
            residual_mask = (route_weight[0] == 0).float()
            route_result_keep = residual_mask * route_h_l1 + route_weight[0] * route_h_l_keep
            result_list[1].append(route_result_keep)
            weights_list[1].append(residual_mask + route_weight[0])
            ## up
            if self.allow_up:
                route_h_l_up = self.res_up(route_h_l[1])
                route_h_l_up = F.interpolate(
                    input=route_h_l_up, scale_factor=2, mode="bilinear", align_corners=False,
                )
                result_list[0].append(route_h_l_up * route_weight[1])
                weights_list[0].append(route_weight[1])
            else:
                result_list[0].append(torch.empty(0))
                weights_list[0].append(torch.empty(0))
            ## down
            if self.allow_down:
                route_h_l_down = self.res_down(route_h_l[-1])
                result_list[2].append(route_h_l_down * route_weight[-1])
                weights_list[2].append(route_weight[-1])
            else:
                result_list[2].append(torch.empty(0))
                weights_list[2].append(torch.empty(0))

            return result_list, weights_list

        h_l = self.cell_ops(h_l1, is_drop_path, drop_prob, layer_rate, step_rate)

        # resolution and dimension change
        # resolution: [up, keep, down]
        # NOTE: origin, using for training
        h_l_keep = self.res_keep(h_l)
        gate_weights_beta_keep = gate_weights_beta[:, 0].unsqueeze(-1)
        # using residual connection if drop cell
        gate_mask = (gate_weights_beta.sum(dim=1, keepdim=True) < 0.0001).float()
        result_list = [[], [gate_mask * h_l1 + gate_weights_beta_keep * h_l_keep], []]
        weights_list = [[], [gate_mask * 1.0 + gate_weights_beta_keep], []]

        if self.allow_up:
            h_l_up = self.res_up(h_l)
            h_l_up = F.interpolate(
                input=h_l_up, scale_factor=2, mode="bilinear", align_corners=False
            )
            gate_weights_beta_up = gate_weights_beta[:, 1].unsqueeze(-1)
            result_list[0].append(h_l_up * gate_weights_beta_up)
            weights_list[0].append(gate_weights_beta_up)

        if self.allow_down:
            h_l_down = self.res_down(h_l)
            gate_weights_beta_down = gate_weights_beta[:, -1].unsqueeze(-1)
            result_list[2].append(h_l_down * gate_weights_beta_down)
            weights_list[2].append(gate_weights_beta_down)

        return result_list, weights_list
