# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
from typing import List, Union

import brt.runtime.distributed as brt_dist
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from tutel_ea.impls import moe_layer_brt as brt_moe
from tutel_ea.impls import moe_layer_brt_dist as brt_dist_moe
from tutel_ea.impls import moe_layer_pt as pt_moe
from tutel_ea.impls import moe_layer_tutel as tutel_moe

_shape_t = Union[int, List[int], torch.Size]


class LayerNorm2D(nn.Module):
    def __init__(self, normalized_shape, norm_layer=None):
        super().__init__()
        self.ln = (
            norm_layer(normalized_shape) if norm_layer is not None else nn.Identity()
        )

    def forward(self, x):
        """
        x: N C H W
        """
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x


class LayerNormFP32(nn.LayerNorm):
    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super(LayerNormFP32, self).__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, in_data: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            return F.layer_norm(
                in_data.float(),
                self.normalized_shape,
                self.weight.float(),
                self.bias.float(),
                self.eps,
            ).type_as(in_data)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        mlpfp32=False,
        mlp_fc2_bias=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlpfp32 = mlpfp32

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=mlp_fc2_bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        if self.mlpfp32:
            with torch.cuda.amp.autocast(enabled=False):
                x = self.fc2.float()(x.type(torch.float32))
                x = self.drop.float()(x)
                # print(f"======>[MLP FP32]")
        else:
            x = self.fc2(x)
            x = self.drop(x)
        return x


class MoEMlp(nn.Module):
    def __init__(
        self,
        model_dim,
        hidden_size,
        num_local_experts,
        top_value,
        capacity_factor=1.25,
        normalize_gate=True,
        batch_prioritized_routing=False,
        vitmoe_loss=False,
        use_global_loss=False,
        use_noise=True,
        moe_drop=0.0,
        init_std=0.02,
        mlpfp32=False,
        mlp_fc2_bias=True,
        moe_post_score=True,
        temperature_init=0.5,
        norm_layer=None,
    ):
        super().__init__()

        self.model_dim = model_dim
        self.hidden_size = hidden_size
        self.num_local_experts = num_local_experts
        self.top_value = top_value
        self.capacity_factor = capacity_factor
        self.normalize_gate = normalize_gate
        self.batch_prioritized_routing = batch_prioritized_routing
        self.init_std = init_std
        self.mlpfp32 = mlpfp32
        self.mlp_fc2_bias = mlp_fc2_bias
        self.moe_post_score = moe_post_score
        self.temperature_init = temperature_init

        self.dist_rank = dist.get_rank()

        MOE_LAYER_VENDOR = os.environ.get("MOE_LAYER_VENDOR", "tutel")
        if MOE_LAYER_VENDOR == "tutel":
            moe = tutel_moe
        elif MOE_LAYER_VENDOR == "pt":
            moe = pt_moe
        elif MOE_LAYER_VENDOR == "brt":
            moe = brt_moe
        elif MOE_LAYER_VENDOR == "brt_dist":
            moe = brt_dist_moe
        else:
            raise ValueError(f"Unknown MOE layer vendor: {MOE_LAYER_VENDOR}")

        self._moe_layer = moe.moe_layer(
            # gate_type='Top%dGate' % top_value,
            gate_type={"type": "top", "k": top_value, "fp32_gate": True},
            model_dim=model_dim,
            experts={
                "type": "ffn",
                "count_per_node": num_local_experts,
                "hidden_size_per_expert": hidden_size,
                "activation_fn": lambda x: F.gelu(x),
                "implicit_dropout_p": moe_drop,
            },
            capacity_factor=capacity_factor,
            # fp32_gate=True,
            scan_expert_func=lambda name, param: setattr(param, "skip_allreduce", True),
            seeds=(1, self.dist_rank + 1, self.dist_rank + 1),
            normalize_gate=normalize_gate,
            batch_prioritized_routing=batch_prioritized_routing,
            vitmoe_loss=vitmoe_loss,
            use_global_loss=use_global_loss,
            use_noise=use_noise,
            mlpfp32=mlpfp32,
            has_fc2_bias=mlp_fc2_bias,
            is_postscore=moe_post_score,
            norm_layer=norm_layer,
            temperature_init=temperature_init,
        )  # todo: ze check .to(device)

        # Distinguish different parameter types: gate, local_experts
        self.local_count = sum(
            [
                torch.numel(param)
                for name, param in self._moe_layer.get_parameter_iterator(
                    param_type="local_experts"
                )
            ]
        )
        self.shared_count = sum(
            [
                torch.numel(param)
                for name, param in self._moe_layer.get_parameter_iterator(
                    param_type="gate"
                )
            ]
        )

    def forward(self, x):
        x = self._moe_layer(x)
        return x, x.l_aux

    def extra_repr(self) -> str:
        return (
            f"[Statistics-{self.dist_rank}] param count for MoE, "
            f"model_dim = {self.model_dim}, hidden_size = {self.hidden_size}, "
            f"num_local_experts = {self.num_local_experts}, top_value = {self.top_value}, "
            f"normalize_gate={self.normalize_gate}, local_experts = {self.local_count}, "
            f"param count for MoE gate = {self.shared_count}."
        )

    def _init_weights(self):
        for expert in self._moe_layer.experts:
            trunc_normal_(expert.fc1_weight, std=self.init_std)
            trunc_normal_(expert.fc2_weight, std=self.init_std)
            nn.init.constant_(expert.fc1_bias, 0)
            if self.mlp_fc2_bias:
                nn.init.constant_(expert.fc2_bias, 0)


class ConvMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        mlpfp32=False,
        proj_ln=False,
        mlp_fc2_bias=True,
    ):
        super().__init__()
        self.mlp = Mlp(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            act_layer=act_layer,
            drop=drop,
            mlpfp32=mlpfp32,
            mlp_fc2_bias=mlp_fc2_bias,
        )
        self.conv_proj = nn.Conv2d(
            in_features,
            in_features,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=False,
            groups=in_features,
        )
        self.proj_ln = LayerNorm2D(in_features, LayerNormFP32) if proj_ln else None

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(L**0.5)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # B C H W
        x = self.conv_proj(x)
        if self.proj_ln:
            x = self.proj_ln(x)  # pylint: disable=not-callable
        x = x.permute(0, 2, 3, 1)  # B H W C
        x = x.reshape(B, L, C)
        x = self.mlp(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        relative_coords_table_type="norm8_log",
        rpe_hidden_dim=512,
        rpe_output_type="normal",
        attn_type="normal",
        mlpfp32=False,
        rm_k_bias=True,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.mlpfp32 = mlpfp32
        self.attn_type = attn_type
        self.rpe_output_type = rpe_output_type
        self.relative_coords_table_type = relative_coords_table_type
        self.rm_k_bias = rm_k_bias

        if self.attn_type == "cosine_mh":
            self.logit_scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
            )
        elif self.attn_type == "normal":
            head_dim = dim // num_heads
            self.scale = qk_scale or head_dim**-0.5
        else:
            raise NotImplementedError()
        if self.relative_coords_table_type != "none":
            # mlp to generate table of relative position bias
            self.rpe_mlp = nn.Sequential(
                nn.Linear(2, rpe_hidden_dim, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(rpe_hidden_dim, num_heads, bias=False),
            )

            # get relative_coords_table
            relative_coords_h = torch.arange(
                -(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32
            )
            relative_coords_w = torch.arange(
                -(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32
            )
            relative_coords_table = (
                torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w]))
                .permute(1, 2, 0)
                .contiguous()
                .unsqueeze(0)
            )  # 1, 2*Wh-1, 2*Ww-1, 2
            if relative_coords_table_type == "linear":
                relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
                relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
            elif relative_coords_table_type == "norm1_log":
                relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
                relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
                relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                    torch.abs(relative_coords_table) + 1.0
                )  # log2
            elif relative_coords_table_type == "norm2_log":
                relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
                relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
                relative_coords_table *= 2  # normalize to -8, 8
                relative_coords_table = (
                    torch.sign(relative_coords_table)
                    * torch.log2(torch.abs(relative_coords_table) + 1.0)
                    / np.log2(3)
                )  # log3
            elif relative_coords_table_type == "norm4_log":
                relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
                relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
                relative_coords_table *= 4  # normalize to -8, 8
                relative_coords_table = (
                    torch.sign(relative_coords_table)
                    * torch.log2(torch.abs(relative_coords_table) + 1.0)
                    / np.log2(5)
                )  # log5
            elif relative_coords_table_type == "norm16_log":
                relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
                relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
                relative_coords_table *= 16  # normalize to -8, 8
                relative_coords_table = (
                    torch.sign(relative_coords_table)
                    * torch.log2(torch.abs(relative_coords_table) + 1.0)
                    / np.log2(17)
                )  # log5
            elif relative_coords_table_type == "norm8_log":
                relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
                relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
                relative_coords_table *= 8  # normalize to -8, 8
                relative_coords_table = (
                    torch.sign(relative_coords_table)
                    * torch.log2(torch.abs(relative_coords_table) + 1.0)
                    / np.log2(8)
                )  # log8
            elif relative_coords_table_type == "norm8_log_192to256":
                if self.window_size[0] == 16:
                    relative_coords_table[:, :, :, 0] /= 11
                    relative_coords_table[:, :, :, 1] /= 11
                elif self.window_size[0] == 8:
                    relative_coords_table[:, :, :, 0] /= 5
                    relative_coords_table[:, :, :, 1] /= 5
                else:
                    raise NotImplementedError
                relative_coords_table *= 8  # normalize to -8, 8
                relative_coords_table = (
                    torch.sign(relative_coords_table)
                    * torch.log2(torch.abs(relative_coords_table) + 1.0)
                    / np.log2(8)
                )  # log8
            else:
                raise NotImplementedError
            self.register_buffer("relative_coords_table", relative_coords_table)
        else:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads
                )
            )  # 2*Wh-1 * 2*Ww-1, nH
            trunc_normal_(self.relative_position_bias_table, std=0.02)
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        if self.rm_k_bias:
            self.qkv = nn.Linear(dim, dim * 3, bias=False)
            if qkv_bias:
                self.q_bias = nn.Parameter(torch.zeros(dim))
                self.v_bias = nn.Parameter(torch.zeros(dim))
            else:
                self.q_bias = None
                self.v_bias = None
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        if self.rm_k_bias:
            qkv_bias = None
            if self.q_bias is not None:
                qkv_bias = torch.cat(
                    (
                        self.q_bias,
                        torch.zeros_like(self.v_bias, requires_grad=False),
                        self.v_bias,
                    )
                )
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        else:
            qkv = (
                self.qkv(x)
                .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        with torch.cuda.amp.autocast(enabled=False):
            if self.attn_type == "cosine_mh":
                q = F.normalize(q.float(), dim=-1)
                k = F.normalize(k.float(), dim=-1)
                logit_scale = torch.clamp(
                    self.logit_scale, max=torch.log(torch.tensor(1.0 / 0.01))
                ).exp()
                attn = (q @ k.transpose(-2, -1)) * logit_scale.float()
            elif self.attn_type == "normal":
                q = q * self.scale
                attn = q.float() @ k.float().transpose(-2, -1)
            else:
                raise NotImplementedError()

        if self.relative_coords_table_type != "none":
            # relative_position_bias_table: 2*Wh-1 * 2*Ww-1, nH
            relative_position_bias_table = self.rpe_mlp(
                self.relative_coords_table
            ).view(-1, self.num_heads)
        else:
            relative_position_bias_table = self.relative_position_bias_table
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        if self.rpe_output_type == "normal":
            pass
        elif self.rpe_output_type == "sigmoid":
            relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        else:
            raise NotImplementedError

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = attn.type_as(x)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        if self.mlpfp32:
            with torch.cuda.amp.autocast(enabled=False):
                x = self.proj.float()(x.type(torch.float32))
                x = self.proj_drop.float()(x)
        else:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlockPost(nn.Module):
    r"""Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        endnorm=False,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        relative_coords_table_type="norm8_log",
        rpe_hidden_dim=512,
        rpe_output_type="normal",
        attn_type="normal",
        mlp_type="normal",
        mlpfp32=False,
        is_moe=False,
        num_local_experts=1,
        top_value=2,
        capacity_factor=1.25,
        normalize_gate=True,
        batch_prioritized_routing=False,
        vitmoe_loss=False,
        use_global_loss=False,
        use_noise=True,
        mlp_fc2_bias=True,
        moe_drop=0.0,
        moe_post_score=True,
        moe_temperature_init=0.5,
        init_std=0.02,
        rm_k_bias=True,
        skip_moe_droppath=False,
        norm_in_moe=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.endnorm = endnorm
        self.mlpfp32 = mlpfp32
        self.is_moe = is_moe
        self.skip_moe_droppath = skip_moe_droppath
        self.norm_in_moe = norm_in_moe

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            relative_coords_table_type=relative_coords_table_type,
            rpe_output_type=rpe_output_type,
            rpe_hidden_dim=rpe_hidden_dim,
            attn_type=attn_type,
            mlpfp32=mlpfp32,
            rm_k_bias=rm_k_bias,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = (
            nn.Identity() if (self.is_moe and self.norm_in_moe) else norm_layer(dim)
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        if self.is_moe:
            self.mlp = MoEMlp(
                model_dim=dim,
                hidden_size=mlp_hidden_dim,
                num_local_experts=num_local_experts,
                top_value=top_value,
                capacity_factor=capacity_factor,
                normalize_gate=normalize_gate,
                batch_prioritized_routing=batch_prioritized_routing,
                vitmoe_loss=vitmoe_loss,
                use_global_loss=use_global_loss,
                use_noise=use_noise,
                moe_drop=moe_drop,
                init_std=init_std,
                mlpfp32=mlpfp32,
                mlp_fc2_bias=mlp_fc2_bias,
                moe_post_score=moe_post_score,
                temperature_init=moe_temperature_init,
                norm_layer=norm_layer if norm_in_moe else None,
            )
        else:
            if mlp_type == "normal":
                self.mlp = Mlp(
                    in_features=dim,
                    hidden_features=mlp_hidden_dim,
                    act_layer=act_layer,
                    drop=drop,
                    mlpfp32=mlpfp32,
                    mlp_fc2_bias=mlp_fc2_bias,
                )
            elif mlp_type == "conv":
                self.mlp = ConvMlp(
                    in_features=dim,
                    hidden_features=mlp_hidden_dim,
                    act_layer=act_layer,
                    drop=drop,
                    mlpfp32=mlpfp32,
                    mlp_fc2_bias=mlp_fc2_bias,
                )
            elif mlp_type == "conv_ln":
                self.mlp = ConvMlp(
                    in_features=dim,
                    hidden_features=mlp_hidden_dim,
                    act_layer=act_layer,
                    drop=drop,
                    mlpfp32=mlpfp32,
                    proj_ln=True,
                    mlp_fc2_bias=mlp_fc2_bias,
                )

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(
                img_mask, self.window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        if self.endnorm:
            self.enorm = norm_layer(dim)
        else:
            self.enorm = None

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert (
            L == H * W
        ), f"input feature has wrong size, with L = {L}, H = {H}, W = {W}"

        shortcut = x

        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        orig_type = x.dtype  # attn may force to fp32
        attn_windows = self.attn(
            x_windows, mask=self.attn_mask
        )  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        if self.mlpfp32:
            with torch.cuda.amp.autocast(enabled=False):
                x = self.norm1.float()(x)
            x = x.type(orig_type)
        else:
            x = self.norm1(x)
        x = shortcut + self.drop_path(x)
        shortcut = x

        orig_type = x.dtype
        if self.is_moe:
            x, l_aux = self.mlp(x)
        else:
            x = self.mlp(x)
            l_aux = torch.zeros(1).to(x.device)
        if self.mlpfp32:
            with torch.cuda.amp.autocast(enabled=False):
                x = self.norm2.float()(x)
            x = x.type(orig_type)
        else:
            x = self.norm2(x)
        if self.skip_moe_droppath and self.is_moe:
            x = shortcut + x
        else:
            x = shortcut + self.drop_path(x)

        if self.endnorm:
            x = self.enorm(x)

        return x, l_aux

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class SwinTransformerBlockPre(nn.Module):
    r"""Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        endnorm=False,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        init_values=-1.0,
        relative_coords_table_type="norm8_log",
        rpe_hidden_dim=512,
        rpe_output_type="normal",
        attn_type="normal",
        mlp_type="normal",
        mlpfp32=False,
        is_moe=False,
        num_local_experts=1,
        top_value=2,
        capacity_factor=1.25,
        normalize_gate=True,
        batch_prioritized_routing=False,
        vitmoe_loss=False,
        use_global_loss=False,
        use_noise=True,
        mlp_fc2_bias=True,
        moe_drop=0.0,
        moe_post_score=True,
        moe_temperature_init=0.5,
        init_std=0.02,
        rm_k_bias=True,
        skip_moe_droppath=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.endnorm = endnorm
        self.mlpfp32 = mlpfp32
        self.init_values = init_values
        self.is_moe = is_moe
        self.skip_moe_droppath = skip_moe_droppath

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            relative_coords_table_type=relative_coords_table_type,
            rpe_output_type=rpe_output_type,
            rpe_hidden_dim=rpe_hidden_dim,
            attn_type=attn_type,
            mlpfp32=mlpfp32,
            rm_k_bias=rm_k_bias,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if self.is_moe:
            self.mlp = MoEMlp(
                model_dim=dim,
                hidden_size=mlp_hidden_dim,
                num_local_experts=num_local_experts,
                top_value=top_value,
                capacity_factor=capacity_factor,
                normalize_gate=normalize_gate,
                batch_prioritized_routing=batch_prioritized_routing,
                vitmoe_loss=vitmoe_loss,
                use_global_loss=use_global_loss,
                use_noise=use_noise,
                moe_drop=moe_drop,
                init_std=init_std,
                mlpfp32=mlpfp32,
                mlp_fc2_bias=mlp_fc2_bias,
                moe_post_score=moe_post_score,
                temperature_init=moe_temperature_init,
            )
        else:
            if mlp_type == "normal":
                self.mlp = Mlp(
                    in_features=dim,
                    hidden_features=mlp_hidden_dim,
                    act_layer=act_layer,
                    drop=drop,
                    mlpfp32=mlpfp32,
                    mlp_fc2_bias=mlp_fc2_bias,
                )
            elif mlp_type == "conv":
                self.mlp = ConvMlp(
                    in_features=dim,
                    hidden_features=mlp_hidden_dim,
                    act_layer=act_layer,
                    drop=drop,
                    mlpfp32=mlpfp32,
                    mlp_fc2_bias=mlp_fc2_bias,
                )
            elif mlp_type == "conv_ln":
                self.mlp = ConvMlp(
                    in_features=dim,
                    hidden_features=mlp_hidden_dim,
                    act_layer=act_layer,
                    drop=drop,
                    mlpfp32=mlpfp32,
                    proj_ln=True,
                    mlp_fc2_bias=mlp_fc2_bias,
                )
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(
                img_mask, self.window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        if init_values >= 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = 1.0, 1.0

        if self.endnorm:
            self.enorm = norm_layer(dim)
        else:
            self.enorm = None

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert (
            L == H * W
        ), f"input feature has wrong size, with L = {L}, H = {H}, W = {W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        orig_type = x.dtype  # attn may force to fp32
        attn_windows = self.attn(
            x_windows, mask=self.attn_mask
        )  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        if self.mlpfp32:
            x = self.gamma_1 * x
            x = x.type(orig_type)
        else:
            x = self.gamma_1 * x
        x = shortcut + self.drop_path(x)
        shortcut = x

        orig_type = x.dtype
        x = self.norm2(x)
        if self.is_moe:
            x, l_aux = self.mlp(x)
            x = self.gamma_2 * x
        else:
            x = self.mlp(x)
            l_aux = torch.zeros(1).to(x.device)
            x = self.gamma_2 * x
        if self.mlpfp32:
            x = x.type(orig_type)
        if self.skip_moe_droppath and self.is_moe:
            x = shortcut + x
        else:
            x = shortcut + self.drop_path(x)

        # if self.is_moe:
        #     print(f"rank: {dist.get_rank()}, drop output: {x.sum()}")
        if self.endnorm:
            x = self.enorm(x)

        return x, l_aux

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}, "
            f"init_values={self.init_values}"
        )

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, postnorm=True):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.postnorm = postnorm

        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim) if postnorm else norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        if self.postnorm:
            x = self.reduction(x)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchReduction1C(nn.Module):
    r"""Patch Reduction Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, postnorm=True):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.postnorm = postnorm

        self.reduction = nn.Linear(dim, dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        if self.postnorm:
            x = self.reduction(x)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += H * W * self.dim * self.dim
        return flops


class ConvPatchMerging(nn.Module):
    r"""Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, postnorm=True):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.postnorm = postnorm

        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=3, stride=2, padding=1)
        self.norm = norm_layer(2 * dim) if postnorm else norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        if self.postnorm:
            x = x.permute(0, 3, 1, 2)  # B C H W
            x = self.reduction(x).flatten(2).transpose(1, 2)  # B H//2*W//2 2*C
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2)  # B C H W
            x = self.reduction(x).flatten(2).transpose(1, 2)  # B H//2*W//2 2*C

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 9 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        init_values (float | None, optional): post norm init value. Default: None
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        checkpoint_blocks=255,
        init_values=-1.0,
        endnorm_interval=-1,
        relative_coords_table_type="norm8_log",
        rpe_hidden_dim=512,
        rpe_output_type="normal",
        attn_type="normal",
        mlp_type="normal",
        mlpfp32_blocks=[-1],
        postnorm=True,
        moe_block=[-1],
        num_local_experts=1,
        top_value=2,
        capacity_factor=1.25,
        normalize_gate=True,
        batch_prioritized_routing=False,
        vitmoe_loss=False,
        use_global_loss=False,
        use_noise=True,
        mlp_fc2_bias=True,
        moe_drop=0.0,
        moe_post_score=True,
        moe_temperature_init=0.5,
        init_std=0.02,
        aux_loss_scale=[1.0],
        rm_k_bias=True,
        skip_moe_droppath=False,
        norm_in_moe=False,
    ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.checkpoint_blocks = checkpoint_blocks
        self.init_values = init_values
        self.endnorm_interval = endnorm_interval
        self.mlpfp32_blocks = mlpfp32_blocks
        self.postnorm = postnorm
        self.norm_in_moe = norm_in_moe  # only work for post-norm
        if len(aux_loss_scale) == 1:
            self.aux_loss_scale = aux_loss_scale * depth
        elif len(aux_loss_scale) == depth:
            self.aux_loss_scale = aux_loss_scale
        else:
            raise NotImplementedError

        # build blocks
        if self.postnorm:
            self.blocks = nn.ModuleList(
                [
                    SwinTransformerBlockPost(
                        dim=dim,
                        input_resolution=input_resolution,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=0 if (i % 2 == 0) else window_size // 2,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[i]
                        if isinstance(drop_path, list)
                        else drop_path,
                        norm_layer=norm_layer,
                        endnorm=True
                        if ((i + 1) % endnorm_interval == 0) and (endnorm_interval > 0)
                        else False,
                        relative_coords_table_type=relative_coords_table_type,
                        rpe_hidden_dim=rpe_hidden_dim,
                        rpe_output_type=rpe_output_type,
                        attn_type=attn_type,
                        mlp_type=mlp_type,
                        mlpfp32=True if i in mlpfp32_blocks else False,
                        is_moe=True if i in moe_block else False,
                        num_local_experts=num_local_experts,
                        top_value=top_value,
                        capacity_factor=capacity_factor,
                        normalize_gate=normalize_gate,
                        batch_prioritized_routing=batch_prioritized_routing,
                        vitmoe_loss=vitmoe_loss,
                        use_global_loss=use_global_loss,
                        use_noise=use_noise,
                        mlp_fc2_bias=mlp_fc2_bias,
                        moe_drop=moe_drop,
                        moe_post_score=moe_post_score,
                        moe_temperature_init=moe_temperature_init,
                        init_std=init_std,
                        rm_k_bias=rm_k_bias,
                        skip_moe_droppath=skip_moe_droppath,
                        norm_in_moe=norm_in_moe,
                    )
                    for i in range(depth)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    SwinTransformerBlockPre(
                        dim=dim,
                        input_resolution=input_resolution,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=0 if (i % 2 == 0) else window_size // 2,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[i]
                        if isinstance(drop_path, list)
                        else drop_path,
                        norm_layer=norm_layer,
                        init_values=init_values[i]
                        if isinstance(init_values, list)
                        else init_values,
                        endnorm=True
                        if ((i + 1) % endnorm_interval == 0) and (endnorm_interval > 0)
                        else False,
                        relative_coords_table_type=relative_coords_table_type,
                        rpe_hidden_dim=rpe_hidden_dim,
                        rpe_output_type=rpe_output_type,
                        attn_type=attn_type,
                        mlp_type=mlp_type,
                        mlpfp32=True if i in mlpfp32_blocks else False,
                        is_moe=True if i in moe_block else False,
                        num_local_experts=num_local_experts,
                        top_value=top_value,
                        capacity_factor=capacity_factor,
                        normalize_gate=normalize_gate,
                        batch_prioritized_routing=batch_prioritized_routing,
                        vitmoe_loss=vitmoe_loss,
                        use_global_loss=use_global_loss,
                        use_noise=use_noise,
                        mlp_fc2_bias=mlp_fc2_bias,
                        moe_drop=moe_drop,
                        moe_post_score=moe_post_score,
                        moe_temperature_init=moe_temperature_init,
                        init_std=init_std,
                        rm_k_bias=rm_k_bias,
                        skip_moe_droppath=skip_moe_droppath,
                    )
                    for i in range(depth)
                ]
            )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer, postnorm=postnorm
            )
        else:
            self.downsample = None

    def forward(self, x):
        l_aux = 0.0
        ckpt_block = 0
        for idx, blk in enumerate(self.blocks):
            if (
                self.use_checkpoint
                and not blk.is_moe
                and ckpt_block < self.checkpoint_blocks
            ):
                x, cur_l_aux = checkpoint.checkpoint(blk, x)
                ckpt_block += 1
            else:
                # print(f"======> forwarding {idx}th block, is_moe: {blk.is_moe}")
                x, cur_l_aux = blk(x)
            l_aux = cur_l_aux * self.aux_loss_scale[idx] + l_aux
        if self.downsample is not None:
            x = self.downsample(x)
        return x, l_aux

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def _init_block_norm_weights(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0.0)
            if blk.is_moe and blk.norm_in_moe:
                for _gate in blk.mlp._moe_layer.gates:
                    nn.init.constant_(_gate.norm.bias, 0)
                    nn.init.constant_(_gate.norm.weight, 0.0)
            else:
                nn.init.constant_(blk.norm2.bias, 0)
                nn.init.constant_(blk.norm2.weight, 0.0)


class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        _B, _C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = (
            Ho
            * Wo
            * self.embed_dim
            * self.in_chans
            * (self.patch_size[0] * self.patch_size[1])
        )
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class ResNetDLNPatchEmbed(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(4)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, 64, 3, stride=2, padding=1, bias=False),
            LayerNorm2D(64, norm_layer),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            LayerNorm2D(64, norm_layer),
            nn.GELU(),
            nn.Conv2d(64, embed_dim, 3, stride=1, padding=1, bias=False),
        )
        self.norm = LayerNorm2D(
            embed_dim, norm_layer if norm_layer is not None else LayerNormFP32
        )  # use ln always
        self.act = nn.GELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        _B, _C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.maxpool(x)
        x = x.flatten(2).transpose(1, 2)
        return x

    def flops(self):
        H, W = self.img_size
        flops = H / 2 * W / 2 * 64 * self.in_chans * (3 * 3)
        flops += H / 2 * W / 2 * 64 * 64 * (3 * 3)
        flops += H / 2 * W / 2 * 64 * self.embed_dim * (3 * 3)
        flops += H / 2 * W / 2 * self.embed_dim * (2**2)
        return flops


class SwinV2TransformerMoE(nn.Module):
    r"""Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        moe_blocks=[[-1], [-1], [-1], [-1]],
        init_values=[1e-5],
        endnorm_interval=-1,
        relative_coords_table_type="norm8_log",
        rpe_hidden_dim=512,
        attn_type="cosine_mh",
        rpe_output_type="sigmoid",
        checkpoint_blocks=[255, 255, 255, 255],
        mlpfp32_layer_blocks=[[-1], [-1], [-1], [-1]],
        postnorm=True,
        mlp_type="normal",
        patch_embed_type="normal",
        patch_merge_type="normal",
        droppath_rule="linear",
        strid16=False,
        strid16_global=False,
        num_local_experts=1,
        top_value=2,
        capacity_factor=1.25,
        normalize_gate=True,
        batch_prioritized_routing=False,
        vitmoe_loss=False,
        use_global_loss=False,
        use_noise=True,
        mlp_fc2_bias=True,
        moe_drop=0.0,
        moe_post_score=True,
        moe_temperature_init=0.5,
        init_std=0.02,
        aux_loss_scales=[[1.0], [1.0], [1.0], [1.0]],
        rm_k_bias=True,
        skip_moe_droppath=False,
        head_2fc=False,
        norm_in_moe=False,
        **kwargs,
    ):
        super().__init__()
        self._ddp_params_and_buffers_to_ignore = list()

        self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        # self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features = (
            int(embed_dim * 2 ** (self.num_layers - 2))
            if strid16
            else int(embed_dim * 2 ** (self.num_layers - 1))
        )
        self.mlp_ratio = mlp_ratio
        self.init_std = init_std
        self.endnorm_interval = endnorm_interval
        self.relative_coords_table_type = relative_coords_table_type
        self.rpe_hidden_dim = rpe_hidden_dim
        self.rpe_output_type = rpe_output_type
        self.attn_type = attn_type
        self.postnorm = postnorm
        self.mlp_type = mlp_type
        self.strid16 = strid16
        self.strid16_global = strid16_global
        self.skip_moe_droppath = skip_moe_droppath
        self.head_2fc = head_2fc
        self.norm_in_moe = norm_in_moe

        # split image into non-overlapping patches
        if patch_embed_type == "normal":
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None,
            )
        elif patch_embed_type == "resnetdln":
            assert patch_size == 4, "check"
            self.patch_embed = ResNetDLNPatchEmbed(
                img_size=img_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                norm_layer=norm_layer,
            )
        elif patch_embed_type == "resnetdnf":
            assert patch_size == 4, "check"
            self.patch_embed = ResNetDLNPatchEmbed(
                img_size=img_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                norm_layer=None,
            )

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=self.init_std)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        if droppath_rule == "linear":
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
            ]  # stochastic depth decay rule
        elif droppath_rule == "2_3_trunc":
            dpr = [
                x.item()
                for x in torch.linspace(0, drop_path_rate, int(2 / 3 * sum(depths)))
            ]
            for _ll in range(int(2 / 3 * sum(depths)), sum(depths)):
                dpr += [dpr[-1]]
        else:
            raise NotImplementedError

        if len(init_values) == 1:
            initv = [init_values[0] for _ in range(sum(depths))]
        elif len(init_values) == 2:
            initv = [
                x.item()
                for x in torch.linspace(init_values[0], init_values[1], sum(depths))
            ]
        else:
            raise NotImplementedError

        if patch_merge_type == "normal":
            downsample_layer = PatchMerging
        elif patch_merge_type == "conv":
            downsample_layer = ConvPatchMerging
        else:
            raise NotImplementedError()
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            cur_dim = (
                int(embed_dim * 2 ** (i_layer - 1))
                if (i_layer == self.num_layers - 1 and strid16)
                else int(embed_dim * 2**i_layer)
            )
            cur_input_resolution = (
                (
                    patches_resolution[0] // (2 ** (i_layer - 1)),
                    patches_resolution[1] // (2 ** (i_layer - 1)),
                )
                if (i_layer == self.num_layers - 1 and strid16)
                else (
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                )
            )
            if i_layer < self.num_layers - 2:
                cur_downsample_layer = downsample_layer
            elif i_layer == self.num_layers - 2:
                if strid16:
                    if depths[-1] == 0:
                        cur_downsample_layer = None
                    else:
                        cur_downsample_layer = PatchReduction1C
                else:
                    cur_downsample_layer = downsample_layer
            else:
                cur_downsample_layer = None

            if i_layer == self.num_layers - 1 and strid16 and strid16_global:
                cur_window_size = window_size * 2
            else:
                cur_window_size = window_size

            layer = BasicLayer(
                # dim=int(embed_dim * 2 ** i_layer),
                dim=cur_dim,
                # input_resolution=(patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)),
                input_resolution=cur_input_resolution,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=cur_window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=cur_downsample_layer,
                use_checkpoint=use_checkpoint,
                checkpoint_blocks=checkpoint_blocks[i_layer],
                init_values=initv[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                endnorm_interval=endnorm_interval,
                relative_coords_table_type=self.relative_coords_table_type,
                rpe_hidden_dim=self.rpe_hidden_dim,
                rpe_output_type=self.rpe_output_type,
                attn_type=self.attn_type,
                mlp_type=self.mlp_type,
                mlpfp32_blocks=mlpfp32_layer_blocks[i_layer],
                postnorm=self.postnorm,
                moe_block=moe_blocks[i_layer],
                num_local_experts=num_local_experts,
                top_value=top_value,
                capacity_factor=capacity_factor,
                normalize_gate=normalize_gate,
                batch_prioritized_routing=batch_prioritized_routing,
                vitmoe_loss=vitmoe_loss,
                use_global_loss=use_global_loss,
                use_noise=use_noise,
                mlp_fc2_bias=mlp_fc2_bias,
                moe_drop=moe_drop,
                moe_post_score=moe_post_score,
                moe_temperature_init=moe_temperature_init,
                init_std=init_std,
                aux_loss_scale=aux_loss_scales[i_layer],
                rm_k_bias=rm_k_bias,
                skip_moe_droppath=skip_moe_droppath,
                norm_in_moe=norm_in_moe,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        if self.head_2fc:
            self.head = (
                nn.Sequential(
                    nn.Linear(self.num_features, self.num_features),
                    nn.ReLU(),
                    nn.Linear(self.num_features, num_classes),
                )
                if num_classes > 0
                else nn.Identity()
            )
        else:
            self.head = (
                nn.Linear(self.num_features, num_classes)
                if num_classes > 0
                else nn.Identity()
            )

        self.apply(self._init_weights)
        # if self.fix_init:
        #     self.fix_init_weight()
        if self.postnorm:
            for bly in self.layers:
                bly._init_block_norm_weights()

    # def fix_init_weight(self):
    #     def rescale(param, layer_id):
    #         param.div_(math.sqrt(2.0 * layer_id))
    #
    #     block_id = 1
    #     for layer in self.layers:
    #         for block in layer.blocks:
    #             rescale(block.attn.proj.weight.data, block_id)
    #
    #             if self.mlp_type == 'normal':
    #                 rescale(block.mlp.fc2.weight.data, block_id)
    #             else:
    #                 rescale(block.mlp.mlp.fc2.weight.data, block_id)
    #             block_id += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, MoEMlp):
            m._init_weights()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {
            "rpe_mlp",
            "logit_scale",
            "relative_position_bias_table",
            "temperature",
            "cosine_projector",
            "sim_matrix",
        }

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        l_aux = 0.0
        for _layer_id, layer in enumerate(self.layers):
            # print(f"=====> forwarding layer {_layer_id}")
            x, cur_l_aux = layer(x)
            l_aux = cur_l_aux + l_aux

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x, l_aux

    def forward(self, x):
        x, l_aux = self.forward_features(x)
        with torch.cuda.amp.autocast(enabled=False):
            x = self.head.float()(x.float())
        return x, l_aux

    def add_param_to_skip_allreduce(self, param_name):
        self._ddp_params_and_buffers_to_ignore.append(param_name)

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for _i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2**self.num_layers)
        )
        flops += self.num_features * self.num_classes
        return flops
