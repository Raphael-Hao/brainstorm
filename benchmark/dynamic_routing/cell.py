# encoding: utf-8
# network file -> build Cell for Dynamic Backbone
# @author: yanwei.li
import brt.frontend.nn as nn
import torch
import torch.nn.functional as F

from ops import OPS, Conv2dNormAct, Identity, kaiming_init_module

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
        self.op_flops = []
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
            self.op_flops.append(op.flops)
        self.real_flops = sum(op_flop for op_flop in self.op_flops)

    def forward(
        self, x, is_drop_path=False, drop_prob=0.0, layer_rate=0.0, step_rate=0.0
    ):
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
                    norm=nn.SyncBatchNorm(2 * self.channel_out),
                    activation=nn.ReLU(),
                ),
            )

            # using Kaiming init
            kaiming_init_module(self.res_down, mode="fan_in")
        if self.allow_up and self.allow_down:
            self.gate_num = 3
        elif self.allow_up or self.allow_down:
            self.gate_num = 2
        else:
            self.gate_num = 1
        if self.using_gate:
            self.gate_conv_beta = nn.Sequential(
                Conv2dNormAct(
                    self.channel_in,
                    self.channel_in // 2,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                    norm=nn.SyncBatchNorm(self.channel_in // 2),
                    activation=nn.ReLU(),
                ),
                nn.AdaptiveAvgPool2d((1, 1)),
                Conv2dNormAct(
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

    def forward(
        self,
        h_l1,
        flops_in_expt=None,
        flops_in_real=None,
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
        if not isinstance(h_l1, float):
            # calculate soft conditional gate
            if self.using_gate:
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
            else:
                gate_weights_beta = self.gate_weights_beta
        else:
            drop_cell = True
        # use for inference
        if not self.training:
            if not drop_cell:
                drop_cell = gate_weights_beta.sum() < 0.0001
            if drop_cell:
                result_list = [[0.0], [h_l1], [0.0]]
                weights_list_beta = [[0.0], [0.0], [0.0]]

            return (
                result_list,
                weights_list_beta,
            )

        h_l = self.cell_ops(h_l1, is_drop_path, drop_prob, layer_rate, step_rate)

        # resolution and dimension change
        # resolution: [up, keep, down]
        h_l_keep = self.res_keep(h_l)
        gate_weights_beta_keep = gate_weights_beta[:, 0].unsqueeze(-1)
        # using residual connection if drop cell
        gate_mask = (gate_weights_beta.sum(dim=1, keepdim=True) < 0.0001).float()
        result_list = [[], [gate_mask * h_l1 + gate_weights_beta_keep * h_l_keep], []]
        weights_list_beta = [[], [gate_mask * 1.0 + gate_weights_beta_keep], []]

        if self.allow_up:
            h_l_up = self.res_up(h_l)
            h_l_up = F.interpolate(
                input=h_l_up, scale_factor=2, mode="bilinear", align_corners=False
            )
            gate_weights_beta_up = gate_weights_beta[:, 1].unsqueeze(-1)
            result_list[0].append(h_l_up * gate_weights_beta_up)
            weights_list_beta[0].append(gate_weights_beta_up)

        if self.allow_down:
            h_l_down = self.res_down(h_l)
            gate_weights_beta_down = gate_weights_beta[:, -1].unsqueeze(-1)
            result_list[2].append(h_l_down * gate_weights_beta_down)
            weights_list_beta[2].append(gate_weights_beta_down)

        return result_list, weights_list_beta
