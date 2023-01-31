import torch
import symbolic
import numpy as np
import sympy
from symbolic.symbolic_tensor import SymbolicTensor
import time

__all_mapping__ = {torch.nn.Embedding : 'torch_embedding',
                   torch.nn.modules.normalization.LayerNorm: "torch_layer_norm",
                   torch.nn.modules.dropout.Dropout: "torch_dropout",
                   torch.nn.modules.linear.Linear: "torch_linear",
                   torch.nn.modules.activation.Tanh: "torch_tanh"}

def proxy_module_forward(target, args, kwargs, module):
    if type(module) in __all_mapping__:
        return getattr(symbolic.symbolic_ops.modules, __all_mapping__[type(module)])(args, kwargs, module, target)
    return None

def torch_embedding(args, kwargs, mod, target):
    input = args[0]
    out = SymbolicTensor(target, (*input.shape, mod.embedding_dim))
    for idx, x in np.ndenumerate(out.data):
        out.data[idx] = input.data[idx[:-1]]
    return out

def torch_layer_norm(args, kwargs, mod, target):
    '''
    LayerNorm. Currently, only support norm on the last dimension
    '''
    input = args[0]
    out = SymbolicTensor(target, input.shape)
    assert input.shape[-1] == mod.normalized_shape[0]
    cached_result = {}
    for idx, x in np.ndenumerate(out.data):
        sum_value = 0
        if idx[:-1] not in cached_result:
            ret = 0
            for last_dim_idx in range(out.shape[-1]):
                ret += input.data[idx[:-1]][last_dim_idx]
            cached_result[idx[:-1]] = ret
        sum_value = cached_result[idx[:-1]]
        out.data[idx] = sum_value
    return out

def torch_dropout(args, kwargs, mod, target):
    input = args[0]
    return input

def torch_tanh(args, kwargs, mod, target):
    input = args[0]
    return input

def torch_linear(args, kwargs, mod, target):
    input = args[0]
    out_shape = list(input.shape)
    out_shape[-1] = mod.out_features
    out_shape = tuple(out_shape)
    out = SymbolicTensor(target, out_shape)

    cached_result = {}
    for idx, x in np.ndenumerate(out.data):
        sum_value = 0
        if idx[:-1] not in cached_result:
            ret = 0
            for last_dim_idx in range(input.shape[-1]):
                ret += input.data[idx[:-1]][last_dim_idx]
            cached_result[idx[:-1]] = ret
        sum_value = cached_result[idx[:-1]]
        out.data[idx] = sum_value
    return out