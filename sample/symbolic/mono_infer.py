import torch
import torch.fx
import traceback

from torch.fx.node import Node, map_aggregate, Target, Argument
from typing import Any, Tuple, NamedTuple, Optional, Dict
from torch.fx._compatibility import compatibility
from torch.fx.passes.shape_prop import TensorMetadata

from symbolic.symbolic_tensor import SymbolicTensor
import numpy as np
from symbolic.symbolic_ops.functions import get_symbolic_func
from symbolic.symbolic_ops.methods import get_symbolic_method
from symbolic.symbolic_ops.modules import proxy_module_forward

import types

__all__ = ['MonoTensorInference']

class MonoTensorInference(torch.fx.Interpreter):
    def call_function(self, target : Target,
                        args : Tuple, kwargs : Dict) -> Any:
        print("call_function", target)
        if all(  not type(v) == SymbolicTensor for v in args):
            out = target(*args, **kwargs)
        else:
            new_func = get_symbolic_func(target)
            if new_func:
                out = new_func(*args, **kwargs)
            else:
                out = target(*args, **kwargs)
        if type(out) == SymbolicTensor:
            out.mono_simplify()
        assert out != None
        return out
        # return super().call_function(n)

    def call_method(self, target : Target,
                    args : Tuple, kwargs : Dict) -> Any:
        print("call_method", target)
        if type(args[0]) == SymbolicTensor:
            new_func = get_symbolic_method(target)
            out = new_func(*args, **kwargs)
        else:
            call_self, *args_tail = args
            if target == 'expand':
                out = call_self.expand(*args_tail, **kwargs)
            elif target == 'dim':
                out = call_self.dim(*args_tail, **kwargs)
            elif target == 'to':
                out = call_self.to(*args_tail, **kwargs)
            elif target == 'size':
                out = call_self.size(*args_tail, **kwargs)
        assert out != None

        if type(out) == SymbolicTensor:
            out.mono_simplify()
        return out

    def call_module(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        print("call_module", target)
        if type(args[0]) == SymbolicTensor:
            mod = self.fetch_attr(target)
            out = proxy_module_forward(target, args, kwargs, mod)
        else:
            out = super().call_module(target, args, kwargs)

        assert out != None
        if type(out) == SymbolicTensor:
            out.mono_simplify()
        return out