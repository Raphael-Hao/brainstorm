from typing import Tuple
from sympy import Symbol
import numpy as np
import torch
import symbolic

def _extract_symbols_from_expr(symbol : Symbol):
    elements = symbol.args
    if len(elements) == 0:
        if type(symbol) == Symbol:
            return [symbol]
        else:
            return []
    ret = []
    for e in elements:
        ret.extend(_extract_symbols_from_expr(e))
    return ret

def _simplify_symbol(symbol : Symbol):
    syms = list(set(_extract_symbols_from_expr(symbol)))
    ret = 0
    for e in syms:
        ret += e
    return ret

class SymbolicTensor:
    def __init__(self, name : str, shape : Tuple[int]):
        self.data = np.empty(shape, dtype=Symbol)
        self.shape = shape
        self.name = name
        self.device = torch.device("cpu")

        for idx, x in np.ndenumerate(self.data):
            self.data[idx] = Symbol(name=f"{self.name}({idx})")

    def size(self):
        return self.shape

    def view(self, new_shape):
        out = SymbolicTensor(self.name+"copy", self.shape)
        out.data = self.data.view().reshape(new_shape)
        out.shape = out.data.shape
        return out

    def permute(self, axes):
        out = SymbolicTensor(self.name+"copy", self.shape)
        out.data = self.data.view().transpose(axes)
        out.shape = out.data.shape
        return out

    def mono_simplify(self):
        apply_simplify = np.vectorize(_simplify_symbol)
        simplified_data = apply_simplify(self.data)
        self.data = simplified_data

    def __add__(self, other):
        return symbolic.symbolic_ops.functions.add(self, other)

    def __mul__(self, other):
        return symbolic.symbolic_ops.functions.multiply(self, other)

    def __truediv__(self, other):
        return symbolic.symbolic_ops.functions.div(self, other)

    def __getitem__(self, idx):
        new_data = self.data[idx]
        if type(new_data) == Symbol:
            out = SymbolicTensor(self.name+f"_at_{idx}", (1))
            out.data = new_data
        else:
            out = SymbolicTensor(self.name+f"_at_{idx}", new_data.shape)
            out.data = new_data
        return out

def symbolify(tensor : torch.Tensor, name : str):
    return SymbolicTensor(name, tensor.shape)