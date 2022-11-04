from torch.fx.node import Node, map_aggregate, Target, Argument

from symbolic.symbolic_tensor import SymbolicTensor
from sympy.functions import exp
import numpy as np
import torch
import symbolic

from symbolic.symbolic_ops.functions import get_symbolic_func

__all_mapping__ = {'neg' : ('func', torch.neg),
                   'size' : ('method', 'size'),
                   'view' : ('method', 'view'),
                   'permute': ('method', 'permute'),
                   'transpose': ('method', 'transpose'),
                   'contiguous': ('method', 'contiguous')}

def get_symbolic_method(target : Target,):
    print("finding method", target, type(target), target in __all_mapping__)
    if target in __all_mapping__:
        if __all_mapping__[target][0] == 'func':
            new_func = get_symbolic_func(__all_mapping__[target][1])
            return new_func
        elif __all_mapping__[target][0] == 'method':
            new_func = getattr(symbolic.symbolic_ops.methods, __all_mapping__[target][1])
            return new_func
    return None

def size(input : SymbolicTensor):
    return input.data.shape

def view(input : SymbolicTensor, shape):
    return input.view(shape)

def permute(input : SymbolicTensor, *axes):
    return input.permute(axes)

def contiguous(input : SymbolicTensor):
    return input

def transpose(input : SymbolicTensor, dim1, dim2):
    if dim1 < 0:
        dim1 = len(input.shape) + dim1
    if dim2 < 0:
        dim2 = len(input.shape) + dim2
    new_axes = [i for i in range(len(input.shape))]
    new_axes[dim1] = dim2
    new_axes[dim2] = dim1
    return permute(input, *new_axes)