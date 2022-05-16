# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch.onnx import register_custom_op_symbolic, symbolic_helper
from torch.onnx.symbolic_helper import parse_args


@parse_args("v", "v", "v")
def scatter_route_symbolic(g, inputs, router_kind, router_num):
    route_results, route_indices, rever_shape = g.op(
        "brt::scatter_route", inputs, router_kind, router_num, outputs=3
    )
    return route_results, route_indices, rever_shape


@parse_args("v", "v", "v", "v", "v")
def gather_route_symbolic(
    g, inputs, reverse_indices, reverse_shape, router_kind, router_num,
):
    return g.op(
        "brt::gather_route",
        inputs,
        reverse_indices,
        reverse_shape,
        router_kind,
        router_num,
    )


register_custom_op_symbolic(
    "brt::symbolic_scatter_route", scatter_route_symbolic, opset_version=9
)
register_custom_op_symbolic(
    "brt::symbolic_gather_route", gather_route_symbolic, opset_version=9
)
