from torch.fx.node import Node, map_aggregate, Target, Argument

from symbolic.symbolic_tensor import SymbolicTensor
from sympy.functions import exp
import numpy as np
import torch
import symbolic
import scipy

__all_mapping__ = {torch.sigmoid : 'sigmoid',
                   torch.neg : 'neg',
                   torch.matmul: 'matmul',
                   torch.add: 'add',
                   torch.div: 'div',
                   torch.nn.functional.softmax: 'softmax',
                   torch._C._nn.gelu: 'gelu'}

def get_symbolic_func(target : Target,):
    if target in __all_mapping__:
        return getattr(symbolic.symbolic_ops.functions, __all_mapping__[target])
    return None

def sigmoid(input : SymbolicTensor):
    out = SymbolicTensor("outxx", input.shape)
    for idx, x in np.ndenumerate(input.data):
        out.data[idx] = 1.0/(1+exp(-x))
    return out

def gelu(input : SymbolicTensor):
    return input

def tanh(input : SymbolicTensor):
    return input

def neg(input : SymbolicTensor):
    out = SymbolicTensor("outxx", input.shape)
    for idx, x in np.ndenumerate(input.data):
        out.data[idx] = -x
    return out

def softmax(a, dim: -1, _stacklevel: 3, dtype: None,):
    apply_exp = np.vectorize(exp)
    exp_value = apply_exp(a.data)
    sum_shape = list([1 for _ in range(len(a.shape))])
    sum_shape[dim] = a.shape[dim]
    sum_shape = tuple(sum_shape)
    c = np.divide(exp_value, np.tile(np.sum(exp_value, axis=dim, keepdims=True), sum_shape))
    out = SymbolicTensor("softmax_out", c.shape)
    out.data = c.view()
    return out

def add(a, b):
    if type(a) == torch.Tensor:
        a = a.numpy()
    if type(b) == torch.Tensor:
        b = b.numpy()
    if type(a) == SymbolicTensor:
        a = a.data
    if type(b) == SymbolicTensor:
        b = b.data
    c = np.add(a, b)
    out = SymbolicTensor("add_out", c.shape)
    out.data = c.view()
    return out

def div(a, b):
    if type(a) == torch.Tensor:
        a = a.numpy()
    if type(b) == torch.Tensor:
        b = b.numpy()
    if type(a) == SymbolicTensor:
        a = a.data
    if type(b) == SymbolicTensor:
        b = b.data
    c = np.add(a, b)
    c = np.divide(a.data, b)
    out = SymbolicTensor("div_out", c.shape)
    out.data = c.view()
    return out

def multiply(a, b):
    if type(a) == torch.Tensor:
        a = a.numpy()
    if type(b) == torch.Tensor:
        b = b.numpy()
    if type(a) == SymbolicTensor:
        a = a.data
    if type(b) == SymbolicTensor:
        b = b.data
    c = np.multiply(a, b)
    if type(c) != np.ndarray:
        c = np.array([c])
    out = SymbolicTensor("mul_out", c.shape)
    out.data = c.view()
    return out

def matmul(a, b):
    if type(a) == torch.Tensor:
        a = a.numpy()
    if type(b) == torch.Tensor:
        b = b.numpy()
    if type(a) == SymbolicTensor:
        a = a.data
    if type(b) == SymbolicTensor:
        b = b.data
    c = np.matmul(a, b)
    out = SymbolicTensor("matmul_out", c.shape)
    out.data = c.view()
    return out